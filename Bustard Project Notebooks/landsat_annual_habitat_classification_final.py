#!/usr/bin/env python
"""Landsat annual habitat classification, Eastern Morocco, HY2008-HY2025.

This is the cleaned production counterpart to the Sentinel-2 workflow.

Key design choices
------------------
- Habitat year: 1 September (year-1) through 31 August (year).
- Annual series: HY2008-HY2025.
- Landsat Collection 2 Tier 1 Level-2 SR from Landsat 5, 7, 8 and 9 is placed
  into six common spectral-band names before compositing.
- The existing fixed five-points-per-polygon reference asset and the saved
  Copernicus GLO-30 terrain asset are reused.
- A Landsat-specific HY2025 predictor stack is materialized once as an Earth
  Engine asset and reused for training, validation sampling and HY2025 mapping.
- At 30 m, multiple fixed points can occupy the same Landsat pixel. The
  Landsat sample table removes pixels with conflicting reference classes and
  retains one deterministic point per remaining 30 m pixel. Thus the same
  raster pixel cannot appear repeatedly across CV folds.
- Local five-fold CV remains grouped by source polygon through ``cv_fold``.
- The operational GEE model uses the agreed deployable RF structure:
  500 trees, 6 variables per split, minimum leaf population 2,
  bag fraction 0.632, seed 42.
- The full six-model tuning table is reproduced locally. Balanced-subsample
  configurations remain sensitivity analyses because GEE Smile Random Forest
  has no directly equivalent class-weight option.
- Large GEE computations are handled as batch exports; no large 500-tree
  confusion matrix is requested interactively.
- Annual maps are materialized as assets one year at a time and then exported
  to Google Drive from those compact stored images.

Important scientific caveat
---------------------------
HY2025 validation does not independently validate historical years. The
pre-2013 series additionally transfers an OLI-era HY2025 classifier to TM/ETM+
data. Check temporal/cross-sensor stability before interpreting persistent
change. An optional OLI-to-ETM+ reflectance transformation is included as a
sensitivity switch but is OFF by default until overlap-year testing is done.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ee
import geemap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


# =============================================================================
# 0. USER CONFIGURATION
# =============================================================================

EE_PROJECT = "rse-global-wetlands"
ASSET_ROOT = "projects/rse-global-wetlands/assets"

# Existing assets from the confirmed HY2025 workflow.
ROI_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_Working_Area"
RF_POINTS_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_RF_Points_HY2025"
TERRAIN_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_Terrain_GLO30"

TRAINING_HABITAT_YEAR = 2025
HABITAT_YEARS = list(range(2008, 2026))

# Optional sensor-space sensitivity. Leave False for the initial production
# run, then compare overlap years before deciding whether harmonization belongs
# in the manuscript series.
APPLY_OLI_TO_ETM_HARMONIZATION = False
SENSOR_SPACE_TAG = "ETMlike" if APPLY_OLI_TO_ETM_HARMONIZATION else "NativeCommon"

# Landsat-specific materialized assets created/reused by this script.
LANDSAT_HY2025_PREDICTOR_ASSET = (
    f"{ASSET_ROOT}/Eastern_Morocco_Landsat_PredictorStack_HY2025_{SENSOR_SPACE_TAG}"
)
LANDSAT_FIXED_SAMPLE_ASSET = (
    f"{ASSET_ROOT}/Eastern_Morocco_Landsat_RF_Samples_Fixed5Pts_HY2025_{SENSOR_SPACE_TAG}"
)
LANDSAT_CLASSIFIER_ASSET = (
    f"{ASSET_ROOT}/Eastern_Morocco_Landsat_RF_Classifier_Fixed5Pts_"
    f"HY2025_T500_{SENSOR_SPACE_TAG}"
)
LANDSAT_TRAINING_PREDICTIONS_ASSET = (
    f"{ASSET_ROOT}/Eastern_Morocco_Landsat_RF_TrainingPredictions_Fixed5Pts_"
    f"HY2025_T500_{SENSOR_SPACE_TAG}"
)

LOCAL_ROOT = Path(
    r"C:\Users\PangY\OneDrive - Smithsonian Institution\Bustard"
    r"\01_Data\Habitat mapping\Annual_Classification\Landsat"
) / SENSOR_SPACE_TAG
ACCURACY_DIR = LOCAL_ROOT / "Accuracy"
MANIFEST_DIR = LOCAL_ROOT / "Manifest"
RF_SAMPLE_DIR = Path(
    r"C:\Users\PangY\OneDrive - Smithsonian Institution\Bustard"
    r"\01_Data\Habitat mapping\RF_Sampling"
)
LOCAL_SAMPLE_CSV = (
    RF_SAMPLE_DIR
    / f"Eastern_Morocco_Landsat_RF_Samples_HY2025_Fixed5Pts_{SENSOR_SPACE_TAG}.csv"
)
LOCAL_SAMPLE_XLSX = (
    RF_SAMPLE_DIR
    / f"Eastern_Morocco_Landsat_RF_Samples_HY2025_Fixed5Pts_{SENSOR_SPACE_TAG}.xlsx"
)

DRIVE_FOLDER = f"Eastern_Morocco_Habitat_Landsat_Annual_{SENSOR_SPACE_TAG}"

# Rebuild switches. Keep False during normal reruns.
REBUILD_TRAINING_PREDICTOR_ASSET = False
REBUILD_FIXED_SAMPLE_ASSET = False
REBUILD_LOCAL_SAMPLE_TABLE = False
RETRAIN_CLASSIFIER = False
REBUILD_GEE_TRAINING_PREDICTIONS = False
OVERWRITE_MAP_ASSETS = False

# Export switches.
EXPORT_MAPS_TO_ASSET = True
EXPORT_MAPS_TO_DRIVE = True
EXPORT_VALID_OBS_TO_ASSET = False
EXPORT_VALID_OBS_TO_DRIVE = False
EXPORT_ACCURACY_EXCEL = True

# Safe task behavior.
WAIT_FOR_CLASSIFIER = True
WAIT_FOR_MAP_ASSET = True
WAIT_FOR_DRIVE_AT_END = False
MAX_ACTIVE_DRIVE_TASKS = 2
POLL_SECONDS = 30

# Landsat Collection 2 Tier 1 Level-2.
LANDSAT_COLLECTIONS = {
    "L5": "LANDSAT/LT05/C02/T1_L2",
    "L7": "LANDSAT/LE07/C02/T1_L2",
    "L8": "LANDSAT/LC08/C02/T1_L2",
    "L9": "LANDSAT/LC09/C02/T1_L2",
}
MAX_SCENE_CLOUD_PERCENT = 80
LANDSAT_SR_SCALE = 0.0000275
LANDSAT_SR_OFFSET = -0.2

COMMON_BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
INDEX_BANDS = ["EVI", "MSAVI", "NDWI", "BSI"]
TERRAIN_BANDS = ["Elevation", "Slope", "Northness", "Eastness"]
PREDICTOR_BANDS = COMMON_BANDS + INDEX_BANDS + TERRAIN_BANDS

CLASS_FIELD = "class_id"
FEATURE_ID_FIELD = "feature_id"
FOLD_FIELD = "cv_fold"
SAMPLE_ID_FIELD = "sample_id"
SAMPLE_PROPERTIES = [SAMPLE_ID_FIELD, FEATURE_ID_FIELD, CLASS_FIELD, FOLD_FIELD]
PIXEL_X_FIELD = "pixel_x"
PIXEL_Y_FIELD = "pixel_y"
PIXEL_KEY_FIELD = "pixel_key"
PIXEL_CLASS_COUNT_FIELD = "pixel_class_count"

CLASS_IDS = list(range(11))
CLASS_MAP = {
    0: "Grass steppe",
    1: "Shrub steppe",
    2: "Wooded steppe",
    3: "Bare rocky",
    4: "Spreading area",
    5: "Salty steppe",
    6: "Dune",
    7: "Wadi and gullies",
    8: "Cultivated field/fallow",
    9: "Water body",
    10: "Built-up",
}

RANDOM_SEED = 42
K_FOLDS = 5
RF_TREES = 500
GEE_VARIABLES_PER_SPLIT = 6
GEE_MIN_LEAF_POPULATION = 2
GEE_BAG_FRACTION = 0.632

EXPORT_CRS = "EPSG:32630"
EXPORT_TRANSFORM = [30, 0, 0, 0, -30, 10_000_000]
SAMPLE_SCALE = 30
MAX_PIXELS = 1e13
DRIVE_NODATA = 255

# Roy et al. OLI -> ETM+ SR sensitivity coefficients: ETM+ = intercept + slope*OLI.
# They are not enabled unless APPLY_OLI_TO_ETM_HARMONIZATION=True.
OLI_TO_ETM_INTERCEPT = [0.0183, 0.0123, 0.0123, 0.0448, 0.0306, 0.0116]
OLI_TO_ETM_SLOPE = [0.8850, 0.9317, 0.9372, 0.8339, 0.8639, 0.9165]

RF_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Baseline": {
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "class_weight": None,
    },
    "Balanced": {
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "class_weight": "balanced_subsample",
    },
    "Leaf2": {
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "class_weight": None,
    },
    "Features6": {
        "max_features": 6,
        "min_samples_leaf": 1,
        "class_weight": None,
    },
    "Features6_Leaf2": {
        "max_features": 6,
        "min_samples_leaf": 2,
        "class_weight": None,
    },
    "Balanced_Features6_Leaf2": {
        "max_features": 6,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
    },
}
MAP_MATCHED_MODEL_NAME = "Features6_Leaf2"
BALANCED_SENSITIVITY_MODEL_NAME = "Balanced_Features6_Leaf2"


# =============================================================================
# 1. LOGGING AND EARTH ENGINE HELPERS
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("landsat_habitat")

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "CANCEL_REQUESTED"}
ACTIVE_STATES = {"READY", "RUNNING"}


def initialize_ee() -> None:
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        LOGGER.info("Earth Engine authentication is required.")
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)
    LOGGER.info("Earth Engine initialized with project '%s'.", EE_PROJECT)


def asset_exists(asset_id: str) -> bool:
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def require_assets(asset_ids: List[str]) -> None:
    missing = [asset_id for asset_id in asset_ids if not asset_exists(asset_id)]
    if missing:
        raise FileNotFoundError(
            "Required Earth Engine asset(s) are missing:\n  " + "\n  ".join(missing)
        )


def delete_asset(asset_id: str) -> None:
    if asset_exists(asset_id):
        LOGGER.warning("Deleting Earth Engine asset: %s", asset_id)
        ee.data.deleteAsset(asset_id)


def wait_for_task(task: ee.batch.Task, label: str) -> Dict[str, Any]:
    previous_state: Optional[str] = None
    while True:
        status = task.status()
        state = str(status.get("state", "UNKNOWN"))
        if state != previous_state:
            LOGGER.info("Task %-48s %s", label, state)
            previous_state = state
        if state in TERMINAL_STATES:
            if state != "COMPLETED":
                raise RuntimeError(
                    f"Earth Engine task failed: {label}\n"
                    f"State: {state}\n"
                    f"Error: {status.get('error_message', '')}"
                )
            return status
        time.sleep(POLL_SECONDS)


def throttle_drive_tasks(tasks: Dict[str, ee.batch.Task]) -> None:
    if MAX_ACTIVE_DRIVE_TASKS <= 0:
        return
    while True:
        active = [
            name
            for name, task in tasks.items()
            if str(task.status().get("state", "UNKNOWN")) in ACTIVE_STATES
        ]
        if len(active) < MAX_ACTIVE_DRIVE_TASKS:
            return
        LOGGER.info("Drive throttle: %d task(s) active; waiting.", len(active))
        time.sleep(POLL_SECONDS)


def wait_for_drive_tasks(tasks: Dict[str, ee.batch.Task]) -> None:
    for name, task in tasks.items():
        state = str(task.status().get("state", "UNKNOWN"))
        if state not in TERMINAL_STATES:
            wait_for_task(task, name)


def habitat_year_dates(year: int) -> Tuple[str, str]:
    return f"{year - 1}-09-01", f"{year}-09-01"


# =============================================================================
# 2. PREFLIGHT QA
# =============================================================================


def preflight_assets() -> None:
    require_assets([ROI_ASSET, RF_POINTS_ASSET, TERRAIN_ASSET])

    points = ee.FeatureCollection(RF_POINTS_ASSET)
    n_points = int(points.size().getInfo())
    properties = ee.Feature(points.first()).propertyNames().getInfo()
    missing_properties = [p for p in SAMPLE_PROPERTIES if p not in properties]
    if missing_properties:
        raise KeyError(
            f"RF point asset is missing required properties: {missing_properties}"
        )
    n_polygons = int(points.aggregate_count_distinct(FEATURE_ID_FIELD).getInfo())
    LOGGER.info("RF points: %d; source polygons: %d.", n_points, n_polygons)
    if n_points != 17_175:
        LOGGER.warning("Expected 17,175 fixed points, but found %d.", n_points)
    if n_polygons != 3_435:
        LOGGER.warning("Expected 3,435 polygons, but found %d.", n_polygons)

    terrain_bands = ee.Image(TERRAIN_ASSET).bandNames().getInfo()
    missing_terrain = [b for b in TERRAIN_BANDS if b not in terrain_bands]
    if missing_terrain:
        raise KeyError(f"Saved terrain asset is missing bands: {missing_terrain}")

    LOGGER.info("Required source assets passed preflight QA.")


# =============================================================================
# 3. LANDSAT PREPROCESSING AND PREDICTORS
# =============================================================================


def clear_mask_l57(image: ee.Image) -> ee.Image:
    """Collection 2 TM/ETM+ QA mask; QA_PIXEL bit 2 is unused."""
    qa = image.select("QA_PIXEL")
    clear = (
        qa.bitwiseAnd(1 << 0).eq(0)
        .And(qa.bitwiseAnd(1 << 1).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )
    return clear.And(image.select("QA_RADSAT").eq(0))


def clear_mask_l89(image: ee.Image) -> ee.Image:
    """Collection 2 OLI/OLI-2 QA mask, including high-confidence cirrus."""
    qa = image.select("QA_PIXEL")
    clear = (
        qa.bitwiseAnd(1 << 0).eq(0)
        .And(qa.bitwiseAnd(1 << 1).eq(0))
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )
    return clear.And(image.select("QA_RADSAT").eq(0))


def prepare_l57(image: ee.Image) -> ee.Image:
    source = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
    reflectance = (
        image.select(source, COMMON_BANDS)
        .multiply(LANDSAT_SR_SCALE)
        .add(LANDSAT_SR_OFFSET)
        .updateMask(clear_mask_l57(image))
    )
    return reflectance.copyProperties(
        image, ["system:time_start", "system:index", "SPACECRAFT_ID", "SENSOR_ID"]
    )


def transform_oli_to_etm(reflectance: ee.Image) -> ee.Image:
    slopes = ee.Image.constant(OLI_TO_ETM_SLOPE).rename(COMMON_BANDS)
    intercepts = ee.Image.constant(OLI_TO_ETM_INTERCEPT).rename(COMMON_BANDS)
    return reflectance.multiply(slopes).add(intercepts).rename(COMMON_BANDS)


def prepare_l89(image: ee.Image) -> ee.Image:
    source = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
    reflectance = (
        image.select(source, COMMON_BANDS)
        .multiply(LANDSAT_SR_SCALE)
        .add(LANDSAT_SR_OFFSET)
        .updateMask(clear_mask_l89(image))
    )
    if APPLY_OLI_TO_ETM_HARMONIZATION:
        reflectance = transform_oli_to_etm(reflectance)
    return reflectance.copyProperties(
        image, ["system:time_start", "system:index", "SPACECRAFT_ID", "SENSOR_ID"]
    )


def build_landsat_collection(
    year: int,
    roi: ee.Geometry,
) -> ee.ImageCollection:
    start_date, end_date = habitat_year_dates(year)
    l5 = (
        ee.ImageCollection(LANDSAT_COLLECTIONS["L5"])
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", MAX_SCENE_CLOUD_PERCENT))
        .map(prepare_l57)
    )
    l7 = (
        ee.ImageCollection(LANDSAT_COLLECTIONS["L7"])
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", MAX_SCENE_CLOUD_PERCENT))
        .map(prepare_l57)
    )
    l8 = (
        ee.ImageCollection(LANDSAT_COLLECTIONS["L8"])
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", MAX_SCENE_CLOUD_PERCENT))
        .map(prepare_l89)
    )
    l9 = (
        ee.ImageCollection(LANDSAT_COLLECTIONS["L9"])
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", MAX_SCENE_CLOUD_PERCENT))
        .map(prepare_l89)
    )
    return l5.merge(l7).merge(l8).merge(l9).sort("system:time_start")


def add_indices(image: ee.Image) -> ee.Image:
    blue = image.select("Blue")
    green = image.select("Green")
    red = image.select("Red")
    nir = image.select("NIR")
    swir1 = image.select("SWIR1")

    evi = image.expression(
        "2.5 * (NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0)",
        {"NIR": nir, "RED": red, "BLUE": blue},
    ).rename("EVI")

    msavi_term = nir.multiply(2).add(1)
    msavi_disc = msavi_term.pow(2).subtract(nir.subtract(red).multiply(8)).max(0)
    msavi = msavi_term.subtract(msavi_disc.sqrt()).divide(2).rename("MSAVI")

    # Do not use normalizedDifference here: scaled Landsat C2 SR can contain
    # negative values and normalizedDifference masks negative inputs.
    ndwi = image.expression(
        "(GREEN - NIR) / (GREEN + NIR)",
        {"GREEN": green, "NIR": nir},
    ).rename("NDWI")

    bsi = image.expression(
        "((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))",
        {"SWIR": swir1, "RED": red, "NIR": nir, "BLUE": blue},
    ).rename("BSI")

    return image.addBands([evi, msavi, ndwi, bsi])


def build_landsat_predictor_stack(
    year: int,
    roi: ee.Geometry,
    terrain: ee.Image,
) -> Tuple[ee.Image, ee.Image, ee.ImageCollection]:
    collection = build_landsat_collection(year, roi)
    composite = collection.median()
    spectral = add_indices(composite)
    valid_obs = collection.select("Blue").count().rename("valid_obs").toUint16()
    start_date, end_date = habitat_year_dates(year)
    stack = (
        spectral.select(COMMON_BANDS + INDEX_BANDS)
        .addBands(terrain)
        .select(PREDICTOR_BANDS)
        .float()
        .set(
            {
                "sensor": "Landsat_5_7_8_9",
                "sensor_space": SENSOR_SPACE_TAG,
                "habitat_year": year,
                "start_date": start_date,
                "end_date_exclusive": end_date,
            }
        )
    )
    return stack, valid_obs, collection


def ensure_training_predictor_asset(
    roi: ee.Geometry,
    terrain: ee.Image,
) -> ee.Image:
    """Materialize/reuse the HY2025 Landsat stack before any RF sampling."""
    if asset_exists(LANDSAT_HY2025_PREDICTOR_ASSET) and not REBUILD_TRAINING_PREDICTOR_ASSET:
        LOGGER.info(
            "Reusing HY2025 Landsat predictor asset: %s",
            LANDSAT_HY2025_PREDICTOR_ASSET,
        )
        return ee.Image(LANDSAT_HY2025_PREDICTOR_ASSET).select(PREDICTOR_BANDS)

    if asset_exists(LANDSAT_HY2025_PREDICTOR_ASSET):
        delete_asset(LANDSAT_HY2025_PREDICTOR_ASSET)

    stack, _, collection = build_landsat_predictor_stack(
        TRAINING_HABITAT_YEAR, roi, terrain
    )
    scene_count = int(collection.size().getInfo())
    if scene_count == 0:
        raise RuntimeError("No Landsat scenes available for HY2025 training stack.")

    task = ee.batch.Export.image.toAsset(
        image=stack.clip(roi),
        description=f"Landsat_PredictorStack_HY2025_{SENSOR_SPACE_TAG}",
        assetId=LANDSAT_HY2025_PREDICTOR_ASSET,
        region=roi,
        crs=EXPORT_CRS,
        crsTransform=EXPORT_TRANSFORM,
        maxPixels=MAX_PIXELS,
        pyramidingPolicy={".default": "mean"},
    )
    task.start()
    LOGGER.info("Started HY2025 Landsat predictor export: %s", task.id)
    wait_for_task(task, "Landsat HY2025 predictor stack")
    return ee.Image(LANDSAT_HY2025_PREDICTOR_ASSET).select(PREDICTOR_BANDS)


# =============================================================================
# 4. LANDSAT-SPECIFIC FIXED-POINT SAMPLE TABLE
# =============================================================================


def build_landsat_fixed_sample_fc(training_stack: ee.Image) -> ee.FeatureCollection:
    """Sample fixed reference points and enforce unique, unambiguous 30 m pixels."""
    points = ee.FeatureCollection(RF_POINTS_ASSET)
    grid_projection = ee.Projection(EXPORT_CRS).atScale(SAMPLE_SCALE)
    pixel_xy = ee.Image.pixelCoordinates(grid_projection).rename(
        [PIXEL_X_FIELD, PIXEL_Y_FIELD]
    )
    sample_image = training_stack.select(PREDICTOR_BANDS).addBands(pixel_xy)

    samples = sample_image.sampleRegions(
        collection=points,
        properties=SAMPLE_PROPERTIES,
        scale=SAMPLE_SCALE,
        projection=grid_projection,
        geometries=False,
        tileScale=4,
    ).filter(
        ee.Filter.notNull(
            PREDICTOR_BANDS
            + [CLASS_FIELD, FEATURE_ID_FIELD, FOLD_FIELD, PIXEL_X_FIELD, PIXEL_Y_FIELD]
        )
    )

    def add_pixel_key(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        x = ee.Number(feature.get(PIXEL_X_FIELD)).format("%.0f")
        y = ee.Number(feature.get(PIXEL_Y_FIELD)).format("%.0f")
        return feature.set(PIXEL_KEY_FIELD, x.cat("_").cat(y))

    keyed = samples.map(add_pixel_key)

    # Identify pixels carrying more than one reference class. Those pixels are
    # ambiguous at 30 m and are excluded rather than arbitrarily assigned.
    unique_pixel_class = keyed.distinct([PIXEL_KEY_FIELD, CLASS_FIELD])
    class_count_by_pixel = ee.Dictionary(
        unique_pixel_class.aggregate_histogram(PIXEL_KEY_FIELD)
    )

    def add_class_count(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        n_classes = ee.Number(class_count_by_pixel.get(feature.get(PIXEL_KEY_FIELD)))
        return feature.set(PIXEL_CLASS_COUNT_FIELD, n_classes)

    unambiguous = keyed.map(add_class_count).filter(
        ee.Filter.eq(PIXEL_CLASS_COUNT_FIELD, 1)
    )

    # Keep one deterministic record per 30 m pixel. Sorting by sample_id makes
    # repeated runs stable and prevents exact pixel duplicates across CV folds.
    return unambiguous.sort(SAMPLE_ID_FIELD).distinct([PIXEL_KEY_FIELD])


def ensure_fixed_sample_asset(training_stack: ee.Image) -> ee.FeatureCollection:
    if asset_exists(LANDSAT_FIXED_SAMPLE_ASSET) and not REBUILD_FIXED_SAMPLE_ASSET:
        LOGGER.info("Reusing Landsat fixed-pixel sample asset: %s", LANDSAT_FIXED_SAMPLE_ASSET)
        return ee.FeatureCollection(LANDSAT_FIXED_SAMPLE_ASSET)

    if asset_exists(LANDSAT_FIXED_SAMPLE_ASSET):
        delete_asset(LANDSAT_FIXED_SAMPLE_ASSET)

    samples = build_landsat_fixed_sample_fc(training_stack)
    task = ee.batch.Export.table.toAsset(
        collection=samples,
        description=f"Landsat_RF_Samples_Fixed5Pts_HY2025_{SENSOR_SPACE_TAG}",
        assetId=LANDSAT_FIXED_SAMPLE_ASSET,
    )
    task.start()
    LOGGER.info("Started Landsat fixed-pixel sample export: %s", task.id)
    wait_for_task(task, "Landsat fixed-pixel sample table")
    return ee.FeatureCollection(LANDSAT_FIXED_SAMPLE_ASSET)


def clean_sample_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = PREDICTOR_BANDS + [CLASS_FIELD, FEATURE_ID_FIELD, FOLD_FIELD]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Landsat RF table is missing required columns: {missing}")

    out = df.copy()
    numeric_columns = (
        PREDICTOR_BANDS
        + [CLASS_FIELD, FEATURE_ID_FIELD, FOLD_FIELD, PIXEL_X_FIELD, PIXEL_Y_FIELD]
    )
    if SAMPLE_ID_FIELD in out.columns:
        numeric_columns.append(SAMPLE_ID_FIELD)
    for col in numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    before = len(out)
    out = out.dropna(subset=required).copy()
    out[CLASS_FIELD] = out[CLASS_FIELD].astype(int)
    out[FEATURE_ID_FIELD] = out[FEATURE_ID_FIELD].astype(int)
    out[FOLD_FIELD] = out[FOLD_FIELD].astype(int)
    for col in [PIXEL_X_FIELD, PIXEL_Y_FIELD]:
        if col in out.columns:
            out[col] = out[col].astype(int)
    if SAMPLE_ID_FIELD in out.columns:
        out[SAMPLE_ID_FIELD] = out[SAMPLE_ID_FIELD].astype("Int64")

    folds = sorted(out[FOLD_FIELD].unique().tolist())
    if folds != list(range(K_FOLDS)):
        raise ValueError(f"Expected cv_fold values 0-{K_FOLDS - 1}; found {folds}")

    if PIXEL_KEY_FIELD in out.columns and out[PIXEL_KEY_FIELD].duplicated().any():
        raise ValueError("Landsat sample asset still contains duplicate 30 m pixel keys.")

    LOGGER.info("Landsat sample QA: %d input rows; %d valid unique-pixel rows.", before, len(out))
    LOGGER.info("Independent source polygons represented: %d", out[FEATURE_ID_FIELD].nunique())
    return out


def load_or_create_local_samples(sample_asset: ee.FeatureCollection) -> pd.DataFrame:
    RF_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    if LOCAL_SAMPLE_CSV.exists() and not REBUILD_LOCAL_SAMPLE_TABLE:
        LOGGER.info("Loading local Landsat RF table: %s", LOCAL_SAMPLE_CSV)
        return clean_sample_dataframe(pd.read_csv(LOCAL_SAMPLE_CSV))

    keep = (
        PREDICTOR_BANDS
        + SAMPLE_PROPERTIES
        + [PIXEL_X_FIELD, PIXEL_Y_FIELD, PIXEL_KEY_FIELD]
    )
    LOGGER.info("Downloading the stored Landsat sample asset once to pandas.")
    df = geemap.ee_to_df(sample_asset.select(keep, retainGeometry=False))
    df = clean_sample_dataframe(df)
    df.to_csv(LOCAL_SAMPLE_CSV, index=False)
    try:
        df.to_excel(LOCAL_SAMPLE_XLSX, index=False)
    except Exception as exc:
        LOGGER.warning("Could not write Landsat sample-table Excel file: %s", exc)
    LOGGER.info("Saved local Landsat sample table: %s", LOCAL_SAMPLE_CSV)
    return df


# =============================================================================
# 5. LOCAL FIVE-FOLD RF TUNING AND ACCURACY
# =============================================================================


def majority_vote(values: pd.Series) -> int:
    counts = values.value_counts()
    winners = counts[counts == counts.max()].index.tolist()
    return int(min(winners))


def evaluate_rf_config(
    samples_df: pd.DataFrame,
    model_name: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    fold_rows: List[Dict[str, Any]] = []
    oof_parts: List[pd.DataFrame] = []
    importance_parts: List[pd.Series] = []

    for fold in range(K_FOLDS):
        train_df = samples_df[samples_df[FOLD_FIELD] != fold].copy()
        valid_df = samples_df[samples_df[FOLD_FIELD] == fold].copy()
        if train_df.empty or valid_df.empty:
            raise ValueError(f"Fold {fold} has an empty training or validation set.")

        model = RandomForestClassifier(
            n_estimators=RF_TREES,
            max_features=config["max_features"],
            min_samples_leaf=config["min_samples_leaf"],
            class_weight=config["class_weight"],
            bootstrap=True,
            random_state=RANDOM_SEED + fold,
            n_jobs=-1,
        )
        model.fit(train_df[PREDICTOR_BANDS], train_df[CLASS_FIELD])
        pred = model.predict(valid_df[PREDICTOR_BANDS]).astype(int)

        _, _, fold_f1, _ = precision_recall_fscore_support(
            valid_df[CLASS_FIELD], pred, labels=CLASS_IDS, zero_division=0
        )
        fold_rows.append(
            {
                "Model": model_name,
                "Fold": fold + 1,
                "N_train": len(train_df),
                "N_validation": len(valid_df),
                "OA": accuracy_score(valid_df[CLASS_FIELD], pred),
                "Kappa": cohen_kappa_score(valid_df[CLASS_FIELD], pred),
                "Balanced_Accuracy": balanced_accuracy_score(valid_df[CLASS_FIELD], pred),
                "Macro_F1": float(np.mean(fold_f1)),
            }
        )

        id_cols = [FEATURE_ID_FIELD, CLASS_FIELD, FOLD_FIELD]
        if SAMPLE_ID_FIELD in valid_df.columns:
            id_cols.insert(0, SAMPLE_ID_FIELD)
        oof = valid_df[id_cols].copy()
        oof["prediction"] = pred
        oof_parts.append(oof)
        importance_parts.append(
            pd.Series(model.feature_importances_, index=PREDICTOR_BANDS, name=f"Fold_{fold + 1}")
        )

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.concat(oof_parts, ignore_index=True)
    y_true = oof_df[CLASS_FIELD].astype(int)
    y_pred = oof_df["prediction"].astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_IDS, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "Model": model_name,
            "class_id": CLASS_IDS,
            "Habitat": [CLASS_MAP[i] for i in CLASS_IDS],
            "N_validation": support,
            "Precision_UserAcc": precision,
            "Recall_ProducerAcc": recall,
            "F1": f1,
        }
    )

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    cm_df = pd.DataFrame(
        cm,
        index=[CLASS_MAP[i] for i in CLASS_IDS],
        columns=[CLASS_MAP[i] for i in CLASS_IDS],
    )
    cm_df.index.name = "Actual"
    cm_df.columns.name = "Predicted"

    polygon_oof = (
        oof_df.groupby([FEATURE_ID_FIELD, CLASS_FIELD], as_index=False)
        .agg(prediction=("prediction", majority_vote), n_points=("prediction", "size"))
    )
    p_precision, p_recall, p_f1, p_support = precision_recall_fscore_support(
        polygon_oof[CLASS_FIELD],
        polygon_oof["prediction"],
        labels=CLASS_IDS,
        zero_division=0,
    )
    polygon_per_class = pd.DataFrame(
        {
            "Model": model_name,
            "class_id": CLASS_IDS,
            "Habitat": [CLASS_MAP[i] for i in CLASS_IDS],
            "N_validation_polygons": p_support,
            "Precision_UserAcc": p_precision,
            "Recall_ProducerAcc": p_recall,
            "F1": p_f1,
        }
    )
    polygon_summary = {
        "Model": model_name,
        "Polygon_OA": accuracy_score(polygon_oof[CLASS_FIELD], polygon_oof["prediction"]),
        "Polygon_Kappa": cohen_kappa_score(polygon_oof[CLASS_FIELD], polygon_oof["prediction"]),
        "Polygon_Balanced_Accuracy": balanced_accuracy_score(
            polygon_oof[CLASS_FIELD], polygon_oof["prediction"]
        ),
        "Polygon_Macro_F1": float(np.mean(p_f1)),
        "N_polygons": len(polygon_oof),
    }

    summary = {
        "Model": model_name,
        "Trees": RF_TREES,
        "MaxFeatures": str(config["max_features"]),
        "MinSamplesLeaf": config["min_samples_leaf"],
        "ClassWeight": str(config["class_weight"]),
        "OA": accuracy_score(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro_F1": float(np.mean(f1)),
        "Fold_OA_Mean": fold_df["OA"].mean(),
        "Fold_OA_SD": fold_df["OA"].std(ddof=1),
        "Fold_Kappa_Mean": fold_df["Kappa"].mean(),
        "Fold_Kappa_SD": fold_df["Kappa"].std(ddof=1),
    }

    importance_matrix = pd.concat(importance_parts, axis=1)
    importance = pd.DataFrame(
        {
            "Model": model_name,
            "Predictor": PREDICTOR_BANDS,
            "Importance_mean": importance_matrix.mean(axis=1).reindex(PREDICTOR_BANDS).values,
            "Importance_SD": importance_matrix.std(axis=1, ddof=1).reindex(PREDICTOR_BANDS).values,
        }
    ).sort_values("Importance_mean", ascending=False)

    return {
        "summary": summary,
        "folds": fold_df,
        "per_class": per_class,
        "confusion": cm_df,
        "oof": oof_df,
        "polygon_oof": polygon_oof,
        "polygon_summary": polygon_summary,
        "polygon_per_class": polygon_per_class,
        "importance": importance,
    }


def run_and_save_cv(samples_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    ACCURACY_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}
    summaries: List[Dict[str, Any]] = []
    folds: List[pd.DataFrame] = []
    per_class: List[pd.DataFrame] = []
    polygon_summaries: List[Dict[str, Any]] = []
    polygon_per_class: List[pd.DataFrame] = []
    importances: List[pd.DataFrame] = []

    for model_name, config in RF_CONFIGS.items():
        LOGGER.info("Running Landsat 5-fold CV: %s", model_name)
        result = evaluate_rf_config(samples_df, model_name, config)
        results[model_name] = result
        summaries.append(result["summary"])
        folds.append(result["folds"])
        per_class.append(result["per_class"])
        polygon_summaries.append(result["polygon_summary"])
        polygon_per_class.append(result["polygon_per_class"])
        importances.append(result["importance"])
        result["confusion"].to_csv(
            ACCURACY_DIR / f"Landsat_CV_ConfusionMatrix_{model_name}.csv"
        )
        result["oof"].to_csv(
            ACCURACY_DIR / f"Landsat_CV_OOF_{model_name}.csv", index=False
        )

    summary_df = pd.DataFrame(summaries).sort_values("Macro_F1", ascending=False)
    folds_df = pd.concat(folds, ignore_index=True)
    per_class_df = pd.concat(per_class, ignore_index=True)
    polygon_summary_df = pd.DataFrame(polygon_summaries).sort_values(
        "Polygon_Macro_F1", ascending=False
    )
    polygon_per_class_df = pd.concat(polygon_per_class, ignore_index=True)
    importance_df = pd.concat(importances, ignore_index=True)

    summary_df.to_csv(
        ACCURACY_DIR / "Landsat_RF_Tuning_5FoldCV_Summary.csv", index=False
    )
    folds_df.to_csv(ACCURACY_DIR / "Landsat_RF_Tuning_5FoldCV_Folds.csv", index=False)
    per_class_df.to_csv(
        ACCURACY_DIR / "Landsat_RF_Tuning_5FoldCV_PerClass.csv", index=False
    )
    polygon_summary_df.to_csv(
        ACCURACY_DIR / "Landsat_RF_Tuning_PolygonMajority_Summary.csv", index=False
    )
    polygon_per_class_df.to_csv(
        ACCURACY_DIR / "Landsat_RF_Tuning_PolygonMajority_PerClass.csv", index=False
    )
    importance_df.to_csv(
        ACCURACY_DIR / "Landsat_RF_Tuning_VariableImportance.csv", index=False
    )

    LOGGER.info("Landsat five-fold tuning table:\n%s", summary_df.round(3).to_string(index=False))
    return summary_df, results


# =============================================================================
# 6. GEE 500-TREE CLASSIFIER AND SAFE DIAGNOSTICS
# =============================================================================


def build_gee_classifier(samples: ee.FeatureCollection) -> ee.Classifier:
    clean = samples.filter(ee.Filter.notNull(PREDICTOR_BANDS + [CLASS_FIELD]))
    return ee.Classifier.smileRandomForest(
        numberOfTrees=RF_TREES,
        variablesPerSplit=GEE_VARIABLES_PER_SPLIT,
        minLeafPopulation=GEE_MIN_LEAF_POPULATION,
        bagFraction=GEE_BAG_FRACTION,
        seed=RANDOM_SEED,
    ).train(
        features=clean,
        classProperty=CLASS_FIELD,
        inputProperties=PREDICTOR_BANDS,
    )


def ensure_classifier_asset(samples: ee.FeatureCollection) -> ee.Classifier:
    if asset_exists(LANDSAT_CLASSIFIER_ASSET) and not RETRAIN_CLASSIFIER:
        LOGGER.info("Reusing Landsat classifier asset: %s", LANDSAT_CLASSIFIER_ASSET)
        return ee.Classifier.load(LANDSAT_CLASSIFIER_ASSET)

    if asset_exists(LANDSAT_CLASSIFIER_ASSET):
        delete_asset(LANDSAT_CLASSIFIER_ASSET)

    classifier = build_gee_classifier(samples)
    task = ee.batch.Export.classifier.toAsset(
        classifier=classifier,
        description=f"Landsat_RF_Classifier_HY2025_T500_{SENSOR_SPACE_TAG}",
        assetId=LANDSAT_CLASSIFIER_ASSET,
    )
    task.start()
    LOGGER.info("Started Landsat 500-tree classifier export: %s", task.id)
    if WAIT_FOR_CLASSIFIER:
        wait_for_task(task, "Landsat 500-tree classifier")
        return ee.Classifier.load(LANDSAT_CLASSIFIER_ASSET)
    LOGGER.warning("Classifier export not awaited; using the in-memory classifier.")
    return classifier


def confusion_metrics_from_df(
    truth: pd.Series,
    prediction: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    cm = confusion_matrix(truth, prediction, labels=CLASS_IDS)
    cm_df = pd.DataFrame(
        cm,
        index=[CLASS_MAP[i] for i in CLASS_IDS],
        columns=[CLASS_MAP[i] for i in CLASS_IDS],
    )
    cm_df.index.name = "Actual"
    cm_df.columns.name = "Predicted"

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=CLASS_IDS, zero_division=0
    )
    class_df = pd.DataFrame(
        {
            "class_id": CLASS_IDS,
            "Habitat": [CLASS_MAP[i] for i in CLASS_IDS],
            "Support": support,
            "Precision_UserAcc": precision,
            "Recall_ProducerAcc": recall,
            "F1": f1,
        }
    )
    metrics = {
        "Training_OA": accuracy_score(truth, prediction),
        "Training_Kappa": cohen_kappa_score(truth, prediction),
        "Training_Balanced_Accuracy": balanced_accuracy_score(truth, prediction),
        "Training_Macro_F1": float(np.mean(f1)),
    }
    return cm_df, class_df, metrics


def ensure_training_prediction_asset(
    classifier: ee.Classifier,
    samples: ee.FeatureCollection,
) -> ee.FeatureCollection:
    if (
        asset_exists(LANDSAT_TRAINING_PREDICTIONS_ASSET)
        and not REBUILD_GEE_TRAINING_PREDICTIONS
    ):
        LOGGER.info(
            "Reusing Landsat GEE training predictions: %s",
            LANDSAT_TRAINING_PREDICTIONS_ASSET,
        )
        return ee.FeatureCollection(LANDSAT_TRAINING_PREDICTIONS_ASSET)

    if asset_exists(LANDSAT_TRAINING_PREDICTIONS_ASSET):
        delete_asset(LANDSAT_TRAINING_PREDICTIONS_ASSET)

    predicted = (
        samples.filter(ee.Filter.notNull(PREDICTOR_BANDS + [CLASS_FIELD]))
        .classify(classifier, "prediction")
        .select(
            [SAMPLE_ID_FIELD, FEATURE_ID_FIELD, FOLD_FIELD, CLASS_FIELD, "prediction"],
            retainGeometry=False,
        )
    )
    task = ee.batch.Export.table.toAsset(
        collection=predicted,
        description=f"Landsat_RF_TrainingPredictions_HY2025_T500_{SENSOR_SPACE_TAG}",
        assetId=LANDSAT_TRAINING_PREDICTIONS_ASSET,
    )
    task.start()
    LOGGER.info("Started batch Landsat training-prediction export: %s", task.id)
    wait_for_task(task, "Landsat GEE training predictions")
    return ee.FeatureCollection(LANDSAT_TRAINING_PREDICTIONS_ASSET)


def get_compact_gee_explain(classifier: ee.Classifier) -> Tuple[float, pd.DataFrame]:
    oob_error = np.nan
    importance_df = pd.DataFrame(columns=["Predictor", "Importance", "Importance_percent"])
    try:
        explain = ee.Dictionary(classifier.explain())
        oob_value = explain.get("outOfBagErrorEstimate").getInfo()
        if oob_value is not None:
            oob_error = float(oob_value)
    except Exception as exc:
        LOGGER.warning("Could not retrieve Landsat GEE OOB error: %s", exc)

    try:
        explain = ee.Dictionary(classifier.explain())
        importance = ee.Dictionary(explain.get("importance")).getInfo()
        if importance:
            importance_df = pd.DataFrame(
                {
                    "Predictor": list(importance.keys()),
                    "Importance": [float(v) for v in importance.values()],
                }
            ).sort_values("Importance", ascending=False)
            total = importance_df["Importance"].sum()
            importance_df["Importance_percent"] = (
                100.0 * importance_df["Importance"] / total if total else np.nan
            )
    except Exception as exc:
        LOGGER.warning("Could not retrieve Landsat GEE variable importance: %s", exc)

    return oob_error, importance_df


def save_gee_diagnostics(
    classifier: ee.Classifier,
    samples: ee.FeatureCollection,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ACCURACY_DIR.mkdir(parents=True, exist_ok=True)
    train_metrics = {
        "Training_OA": np.nan,
        "Training_Kappa": np.nan,
        "Training_Balanced_Accuracy": np.nan,
        "Training_Macro_F1": np.nan,
    }
    cm_df = pd.DataFrame()
    class_df = pd.DataFrame()

    try:
        pred_asset = ensure_training_prediction_asset(classifier, samples)
        pred_df = geemap.ee_to_df(
            pred_asset.select(
                [SAMPLE_ID_FIELD, FEATURE_ID_FIELD, FOLD_FIELD, CLASS_FIELD, "prediction"],
                retainGeometry=False,
            )
        )
        pred_df[CLASS_FIELD] = pd.to_numeric(pred_df[CLASS_FIELD], errors="coerce")
        pred_df["prediction"] = pd.to_numeric(pred_df["prediction"], errors="coerce")
        pred_df = pred_df.dropna(subset=[CLASS_FIELD, "prediction"]).copy()
        pred_df[CLASS_FIELD] = pred_df[CLASS_FIELD].astype(int)
        pred_df["prediction"] = pred_df["prediction"].astype(int)
        pred_df.to_csv(
            ACCURACY_DIR / "Landsat_GEE_TrainingPredictions.csv", index=False
        )

        cm_df, class_df, train_metrics = confusion_metrics_from_df(
            pred_df[CLASS_FIELD], pred_df["prediction"]
        )
        cm_df.to_csv(ACCURACY_DIR / "Landsat_GEE_Training_ConfusionMatrix.csv")
        class_df.to_csv(
            ACCURACY_DIR / "Landsat_GEE_Training_PerClass.csv", index=False
        )
    except Exception as exc:
        LOGGER.warning(
            "Landsat GEE resubstitution diagnostics failed; annual maps will continue: %s",
            exc,
        )

    oob_error, importance_df = get_compact_gee_explain(classifier)
    if not importance_df.empty:
        importance_df.to_csv(
            ACCURACY_DIR / "Landsat_GEE_VariableImportance.csv", index=False
        )

    summary = pd.DataFrame(
        [
            {
                "Model": "GEE_SmileRF_Features6_Leaf2",
                "SensorSpace": SENSOR_SPACE_TAG,
                "Trees": RF_TREES,
                "VariablesPerSplit": GEE_VARIABLES_PER_SPLIT,
                "MinLeafPopulation": GEE_MIN_LEAF_POPULATION,
                "BagFraction": GEE_BAG_FRACTION,
                **train_metrics,
                "OOB_Error": oob_error,
                "Approx_OOB_Accuracy": 1.0 - oob_error if np.isfinite(oob_error) else np.nan,
            }
        ]
    )
    summary.to_csv(
        ACCURACY_DIR / "Landsat_GEE_Accuracy_Summary.csv", index=False
    )
    return summary, cm_df, class_df, importance_df


# =============================================================================
# 7. CONSOLIDATED ACCURACY OUTPUTS
# =============================================================================


def save_accuracy_overview(
    cv_summary: pd.DataFrame,
    cv_results: Dict[str, Dict[str, Any]],
    gee_summary: pd.DataFrame,
    gee_cm: pd.DataFrame,
    gee_per_class: pd.DataFrame,
    gee_importance: pd.DataFrame,
) -> pd.DataFrame:
    map_row = cv_summary.loc[cv_summary["Model"] == MAP_MATCHED_MODEL_NAME].iloc[0]
    balanced_row = cv_summary.loc[
        cv_summary["Model"] == BALANCED_SENSITIVITY_MODEL_NAME
    ].iloc[0]
    polygon_row = cv_results[MAP_MATCHED_MODEL_NAME]["polygon_summary"]
    gee_row = gee_summary.iloc[0]

    overview = pd.DataFrame(
        [
            {
                "Evaluation": "5-fold polygon-grouped CV (map-matched)",
                "Model": MAP_MATCHED_MODEL_NAME,
                "OA": map_row["OA"],
                "Kappa": map_row["Kappa"],
                "Balanced_Accuracy": map_row["Balanced_Accuracy"],
                "Macro_F1": map_row["Macro_F1"],
                "Role": "Primary HY2025 validation for Landsat map model",
            },
            {
                "Evaluation": "5-fold polygon-grouped CV (balanced sensitivity)",
                "Model": BALANCED_SENSITIVITY_MODEL_NAME,
                "OA": balanced_row["OA"],
                "Kappa": balanced_row["Kappa"],
                "Balanced_Accuracy": balanced_row["Balanced_Accuracy"],
                "Macro_F1": balanced_row["Macro_F1"],
                "Role": "Class-imbalance sensitivity; not directly deployable in GEE",
            },
            {
                "Evaluation": "Polygon-majority 5-fold CV (map-matched)",
                "Model": MAP_MATCHED_MODEL_NAME,
                "OA": polygon_row["Polygon_OA"],
                "Kappa": polygon_row["Polygon_Kappa"],
                "Balanced_Accuracy": polygon_row["Polygon_Balanced_Accuracy"],
                "Macro_F1": polygon_row["Polygon_Macro_F1"],
                "Role": "Complementary source-polygon validation",
            },
            {
                "Evaluation": "GEE training/resubstitution",
                "Model": "GEE_SmileRF_Features6_Leaf2",
                "OA": gee_row["Training_OA"],
                "Kappa": gee_row["Training_Kappa"],
                "Balanced_Accuracy": gee_row["Training_Balanced_Accuracy"],
                "Macro_F1": gee_row["Training_Macro_F1"],
                "Role": "Internal diagnostic; not independent validation",
            },
            {
                "Evaluation": "GEE out-of-bag",
                "Model": "GEE_SmileRF_Features6_Leaf2",
                "OA": gee_row["Approx_OOB_Accuracy"],
                "Kappa": np.nan,
                "Balanced_Accuracy": np.nan,
                "Macro_F1": np.nan,
                "Role": "Internal sample-level diagnostic",
            },
        ]
    )
    overview.to_csv(ACCURACY_DIR / "Landsat_Accuracy_Overview.csv", index=False)

    if EXPORT_ACCURACY_EXCEL:
        workbook = ACCURACY_DIR / f"Landsat_Accuracy_Tables_HY2025_{SENSOR_SPACE_TAG}.xlsx"
        try:
            with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
                overview.to_excel(writer, sheet_name="Accuracy_overview", index=False)
                cv_summary.to_excel(writer, sheet_name="CV_tuning_summary", index=False)
                pd.concat(
                    [r["folds"] for r in cv_results.values()], ignore_index=True
                ).to_excel(writer, sheet_name="CV_fold_metrics", index=False)
                pd.concat(
                    [r["per_class"] for r in cv_results.values()], ignore_index=True
                ).to_excel(writer, sheet_name="CV_per_class", index=False)
                pd.DataFrame(
                    [r["polygon_summary"] for r in cv_results.values()]
                ).to_excel(writer, sheet_name="Polygon_summary", index=False)
                cv_results[MAP_MATCHED_MODEL_NAME]["confusion"].to_excel(
                    writer, sheet_name="CM_map_matched"
                )
                cv_results[BALANCED_SENSITIVITY_MODEL_NAME]["confusion"].to_excel(
                    writer, sheet_name="CM_balanced"
                )
                gee_summary.to_excel(writer, sheet_name="GEE_summary", index=False)
                if not gee_per_class.empty:
                    gee_per_class.to_excel(writer, sheet_name="GEE_per_class", index=False)
                if not gee_cm.empty:
                    gee_cm.to_excel(writer, sheet_name="GEE_train_CM")
                if not gee_importance.empty:
                    gee_importance.to_excel(
                        writer, sheet_name="GEE_importance", index=False
                    )
            LOGGER.info("Saved Landsat accuracy workbook: %s", workbook)
        except Exception as exc:
            LOGGER.warning("Could not write Landsat accuracy Excel workbook: %s", exc)

    return overview


# =============================================================================
# 8. ANNUAL LANDSAT CLASSIFICATION AND EXPORTS
# =============================================================================


def annual_map_asset_id(year: int) -> str:
    return (
        f"{ASSET_ROOT}/Eastern_Morocco_Habitat_Landsat_RF_Fixed5Pts_"
        f"HY{year}_T500_{SENSOR_SPACE_TAG}"
    )


def annual_valid_obs_asset_id(year: int) -> str:
    return f"{ASSET_ROOT}/Eastern_Morocco_Landsat_ValidObs_HY{year}_{SENSOR_SPACE_TAG}"


def get_year_stack(
    year: int,
    roi: ee.Geometry,
    terrain: ee.Image,
) -> Tuple[ee.Image, ee.Image, ee.ImageCollection]:
    if year == TRAINING_HABITAT_YEAR:
        stack = ee.Image(LANDSAT_HY2025_PREDICTOR_ASSET).select(PREDICTOR_BANDS)
        collection = build_landsat_collection(year, roi)
        valid_obs = collection.select("Blue").count().rename("valid_obs").toUint16()
        return stack, valid_obs, collection
    return build_landsat_predictor_stack(year, roi, terrain)


def classify_year(
    year: int,
    classifier: ee.Classifier,
    roi: ee.Geometry,
    terrain: ee.Image,
) -> Tuple[ee.Image, ee.Image, ee.ImageCollection]:
    stack, valid_obs, collection = get_year_stack(year, roi, terrain)
    start_date, end_date = habitat_year_dates(year)
    classified = (
        stack.select(PREDICTOR_BANDS)
        .classify(classifier)
        .rename("class_id")
        .toUint8()
        .clip(roi)
        .set(
            {
                "sensor": "Landsat_5_7_8_9",
                "sensor_space": SENSOR_SPACE_TAG,
                "habitat_year": year,
                "start_date": start_date,
                "end_date_exclusive": end_date,
                "training_habitat_year": TRAINING_HABITAT_YEAR,
                "classifier_asset": LANDSAT_CLASSIFIER_ASSET,
                "trees": RF_TREES,
                "variables_per_split": GEE_VARIABLES_PER_SPLIT,
                "min_leaf_population": GEE_MIN_LEAF_POPULATION,
                "bag_fraction": GEE_BAG_FRACTION,
            }
        )
    )
    return classified, valid_obs.clip(roi), collection


def ensure_map_asset(year: int, classified: ee.Image, roi: ee.Geometry) -> Tuple[ee.Image, str]:
    asset_id = annual_map_asset_id(year)
    if not EXPORT_MAPS_TO_ASSET:
        return classified, "not_requested"

    if asset_exists(asset_id) and not OVERWRITE_MAP_ASSETS:
        LOGGER.info("Reusing annual Landsat map asset: %s", asset_id)
        return ee.Image(asset_id), "existing"

    if asset_exists(asset_id):
        delete_asset(asset_id)

    task = ee.batch.Export.image.toAsset(
        image=classified,
        description=f"Eastern_Morocco_Habitat_Landsat_RF_HY{year}_{SENSOR_SPACE_TAG}",
        assetId=asset_id,
        region=roi,
        crs=EXPORT_CRS,
        crsTransform=EXPORT_TRANSFORM,
        maxPixels=MAX_PIXELS,
        pyramidingPolicy={".default": "mode"},
    )
    task.start()
    LOGGER.info("Started Landsat map asset HY%d: %s", year, task.id)
    if WAIT_FOR_MAP_ASSET:
        wait_for_task(task, f"Landsat map asset HY{year}")
        return ee.Image(asset_id), "completed"
    return classified, "submitted"


def ensure_valid_obs_asset(year: int, valid_obs: ee.Image, roi: ee.Geometry) -> str:
    if not EXPORT_VALID_OBS_TO_ASSET:
        return "not_requested"
    asset_id = annual_valid_obs_asset_id(year)
    if asset_exists(asset_id) and not OVERWRITE_MAP_ASSETS:
        return "existing"
    if asset_exists(asset_id):
        delete_asset(asset_id)
    task = ee.batch.Export.image.toAsset(
        image=valid_obs.toUint16(),
        description=f"Eastern_Morocco_Landsat_ValidObs_HY{year}_{SENSOR_SPACE_TAG}",
        assetId=asset_id,
        region=roi,
        crs=EXPORT_CRS,
        crsTransform=EXPORT_TRANSFORM,
        maxPixels=MAX_PIXELS,
        pyramidingPolicy={".default": "mean"},
    )
    task.start()
    wait_for_task(task, f"Landsat valid observations HY{year}")
    return "completed"


def start_drive_export(
    year: int,
    map_image: ee.Image,
    roi: ee.Geometry,
    drive_tasks: Dict[str, ee.batch.Task],
) -> Optional[ee.batch.Task]:
    if not EXPORT_MAPS_TO_DRIVE:
        return None
    throttle_drive_tasks(drive_tasks)
    name = f"Eastern_Morocco_Habitat_Landsat_RF_HY{year}_T500_{SENSOR_SPACE_TAG}"
    drive_image = (
        map_image.unmask(value=DRIVE_NODATA, sameFootprint=False)
        .clip(roi)
        .toUint8()
    )
    task = ee.batch.Export.image.toDrive(
        image=drive_image,
        description=f"{name}_Drive",
        folder=DRIVE_FOLDER,
        fileNamePrefix=name,
        region=roi,
        crs=EXPORT_CRS,
        crsTransform=EXPORT_TRANSFORM,
        maxPixels=MAX_PIXELS,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True, "noData": DRIVE_NODATA},
        skipEmptyTiles=True,
    )
    task.start()
    drive_tasks[name] = task
    LOGGER.info("Started Landsat Google Drive export HY%d: %s", year, task.id)
    return task


def start_valid_obs_drive_export(
    year: int,
    valid_obs: ee.Image,
    roi: ee.Geometry,
    drive_tasks: Dict[str, ee.batch.Task],
) -> Optional[ee.batch.Task]:
    if not EXPORT_VALID_OBS_TO_DRIVE:
        return None
    throttle_drive_tasks(drive_tasks)
    name = f"Eastern_Morocco_Landsat_ValidObs_HY{year}_{SENSOR_SPACE_TAG}"
    task = ee.batch.Export.image.toDrive(
        image=valid_obs.toUint16(),
        description=f"{name}_Drive",
        folder=DRIVE_FOLDER,
        fileNamePrefix=name,
        region=roi,
        crs=EXPORT_CRS,
        crsTransform=EXPORT_TRANSFORM,
        maxPixels=MAX_PIXELS,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
        skipEmptyTiles=True,
    )
    task.start()
    drive_tasks[name] = task
    return task


def run_annual_exports(
    classifier: ee.Classifier,
    roi: ee.Geometry,
    terrain: ee.Image,
) -> pd.DataFrame:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    drive_tasks: Dict[str, ee.batch.Task] = {}
    rows: List[Dict[str, Any]] = []

    for year in HABITAT_YEARS:
        LOGGER.info("=== Landsat HY%d ===", year)
        start_date, end_date = habitat_year_dates(year)
        collection = build_landsat_collection(year, roi)
        scene_count = int(collection.size().getInfo())
        if scene_count == 0:
            LOGGER.warning("HY%d has no filtered Landsat scenes; skipping.", year)
            rows.append(
                {
                    "habitat_year": year,
                    "start_date": start_date,
                    "end_date_exclusive": end_date,
                    "scene_count": 0,
                    "status": "skipped_no_scenes",
                }
            )
            continue

        try:
            sensor_counts = collection.aggregate_histogram("SPACECRAFT_ID").getInfo()
        except Exception:
            sensor_counts = {}

        classified, valid_obs, _ = classify_year(year, classifier, roi, terrain)
        map_image, asset_status = ensure_map_asset(year, classified, roi)
        obs_status = ensure_valid_obs_asset(year, valid_obs, roi)
        drive_task = start_drive_export(year, map_image, roi, drive_tasks)
        obs_drive_task = start_valid_obs_drive_export(year, valid_obs, roi, drive_tasks)

        row = {
            "sensor": "Landsat",
            "sensor_space": SENSOR_SPACE_TAG,
            "habitat_year": year,
            "start_date": start_date,
            "end_date_exclusive": end_date,
            "scene_count": scene_count,
            "sensor_scene_counts": json.dumps(sensor_counts, sort_keys=True),
            "map_asset": annual_map_asset_id(year) if EXPORT_MAPS_TO_ASSET else "",
            "map_asset_status": asset_status,
            "drive_folder": DRIVE_FOLDER if EXPORT_MAPS_TO_DRIVE else "",
            "drive_task_id": drive_task.id if drive_task is not None else "",
            "valid_obs_asset_status": obs_status,
            "valid_obs_drive_task_id": obs_drive_task.id if obs_drive_task else "",
            "training_habitat_year": TRAINING_HABITAT_YEAR,
            "validation_scope": (
                "HY2025 polygon-grouped 5-fold CV; historical temporal/cross-sensor transfer"
            ),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(
            MANIFEST_DIR / "Landsat_Annual_Export_Manifest.csv", index=False
        )

    if WAIT_FOR_DRIVE_AT_END:
        wait_for_drive_tasks(drive_tasks)

    task_states = {
        name: str(task.status().get("state", "UNKNOWN"))
        for name, task in drive_tasks.items()
    }
    manifest = pd.DataFrame(rows)
    if not manifest.empty and "habitat_year" in manifest.columns:
        manifest["drive_task_state_at_script_end"] = manifest["habitat_year"].map(
            lambda y: task_states.get(
                f"Eastern_Morocco_Habitat_Landsat_RF_HY{int(y)}_T500_{SENSOR_SPACE_TAG}",
                "",
            )
        )
    manifest.to_csv(MANIFEST_DIR / "Landsat_Annual_Export_Manifest.csv", index=False)
    return manifest


# =============================================================================
# 9. RUN CONFIGURATION RECORD
# =============================================================================


def save_run_configuration() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "ee_project": EE_PROJECT,
        "roi_asset": ROI_ASSET,
        "rf_points_asset": RF_POINTS_ASSET,
        "terrain_asset": TERRAIN_ASSET,
        "hy2025_predictor_asset": LANDSAT_HY2025_PREDICTOR_ASSET,
        "fixed_sample_asset": LANDSAT_FIXED_SAMPLE_ASSET,
        "classifier_asset": LANDSAT_CLASSIFIER_ASSET,
        "training_prediction_asset": LANDSAT_TRAINING_PREDICTIONS_ASSET,
        "local_sample_csv": str(LOCAL_SAMPLE_CSV),
        "habitat_years": HABITAT_YEARS,
        "sensor_space": SENSOR_SPACE_TAG,
        "apply_oli_to_etm_harmonization": APPLY_OLI_TO_ETM_HARMONIZATION,
        "predictor_bands": PREDICTOR_BANDS,
        "rf_trees": RF_TREES,
        "variables_per_split": GEE_VARIABLES_PER_SPLIT,
        "min_leaf_population": GEE_MIN_LEAF_POPULATION,
        "bag_fraction": GEE_BAG_FRACTION,
        "random_seed": RANDOM_SEED,
        "export_crs": EXPORT_CRS,
        "export_transform": EXPORT_TRANSFORM,
        "landsat_sr_scale": LANDSAT_SR_SCALE,
        "landsat_sr_offset": LANDSAT_SR_OFFSET,
        "class_map": CLASS_MAP,
        "warning": (
            "HY2025 validation does not independently validate historical years; "
            "evaluate temporal and cross-sensor transfer before change inference."
        ),
    }
    with (MANIFEST_DIR / "Landsat_Run_Configuration.json").open(
        "w", encoding="utf-8"
    ) as fp:
        json.dump(config, fp, indent=2)


# =============================================================================
# 10. MAIN
# =============================================================================


def main() -> None:
    initialize_ee()
    ACCURACY_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    RF_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    preflight_assets()
    roi = ee.FeatureCollection(ROI_ASSET).geometry()
    terrain = ee.Image(TERRAIN_ASSET).select(TERRAIN_BANDS).resample("bilinear")

    if APPLY_OLI_TO_ETM_HARMONIZATION:
        LOGGER.warning(
            "OLI-to-ETM+ harmonization is ON. Treat this as a sensitivity version "
            "until overlap-year testing confirms it improves temporal consistency."
        )
    else:
        LOGGER.warning(
            "Landsat NativeCommon sensor space is being used. Inspect cross-sensor "
            "stability before interpreting long-term transitions."
        )

    # 1) Build/reuse a stable HY2025 Landsat predictor asset.
    training_stack = ensure_training_predictor_asset(roi, terrain)

    # 2) Build/reuse the unique 30 m fixed-point sample table and cache locally.
    sample_asset = ensure_fixed_sample_asset(training_stack)
    samples_df = load_or_create_local_samples(sample_asset)

    # 3) Reproduce the complete six-model five-fold tuning table.
    cv_summary, cv_results = run_and_save_cv(samples_df)

    # 4) Train/reuse the 500-tree deployable GEE classifier.
    classifier = ensure_classifier_asset(sample_asset)

    # 5) Obtain safe training/OOB diagnostics.
    gee_summary, gee_cm, gee_per_class, gee_importance = save_gee_diagnostics(
        classifier, sample_asset
    )

    # 6) Consolidated accuracy tables/workbook.
    overview = save_accuracy_overview(
        cv_summary,
        cv_results,
        gee_summary,
        gee_cm,
        gee_per_class,
        gee_importance,
    )
    LOGGER.info("Landsat accuracy overview:\n%s", overview.round(3).to_string(index=False))

    # 7) Produce HY2008-HY2025 maps.
    manifest = run_annual_exports(classifier, roi, terrain)
    save_run_configuration()

    LOGGER.info("Landsat production workflow finished.")
    LOGGER.info("Accuracy tables: %s", ACCURACY_DIR)
    LOGGER.info(
        "Export manifest: %s", MANIFEST_DIR / "Landsat_Annual_Export_Manifest.csv"
    )
    LOGGER.info("Google Drive folder: %s", DRIVE_FOLDER)
    if not WAIT_FOR_DRIVE_AT_END and EXPORT_MAPS_TO_DRIVE:
        LOGGER.info("Drive tasks continue on Earth Engine after the Python script exits.")


if __name__ == "__main__":
    main()
