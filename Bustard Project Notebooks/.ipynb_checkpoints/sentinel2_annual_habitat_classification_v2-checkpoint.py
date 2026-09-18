#!/usr/bin/env python
"""
Production Sentinel-2 annual habitat mapping, Eastern Morocco
==============================================================

Purpose
-------
Generate annual habitat maps for 2018-2025 while minimizing repeated
Earth Engine computation.

Default temporal window
-----------------------
Apr-Aug:
    April 1 -> September 1 (exclusive) of each target year.

The temporal window is configurable with TEMPORAL_WINDOW:
    "Mar_Aug", "Apr_Aug", "May_Aug", or "Full_HY".

Core design for low GEE load
----------------------------
1. REUSE the existing fixed 17,175 training points.
2. REUSE the existing static terrain asset.
3. MATERIALIZE each annual Sentinel-2 spectral/index composite ONCE as a
   persistent Earth Engine image asset.
4. MATERIALIZE the HY2025 17,175-row training sample table ONCE from the
   stored HY2025 spectral asset + stored terrain asset.
5. TRAIN and EXPORT the GEE classifier ONCE.
6. For each year, classify from the STORED annual spectral asset + STORED
   terrain asset.
7. MATERIALIZE each habitat map as an EE asset before optionally exporting
   it to Google Drive.
8. Existing assets are reused automatically on reruns.

This script intentionally does NOT rerun local 5-fold CV. The temporal-window
screening has already been performed separately. Final accuracy analysis can
be run independently without rebuilding the annual maps.

Important
---------
The older HY2025 classifier/predictor assets should NOT be reused if they were
built from a different temporal-window definition. This script therefore uses
new asset names that include the chosen window tag.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ee
import pandas as pd


# =============================================================================
# 0. USER CONFIGURATION
# =============================================================================

EE_PROJECT = "rse-global-wetlands"
ASSET_ROOT = "projects/rse-global-wetlands/assets"

# Existing authoritative assets that ARE reused.
ROI_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_Working_Area"
RF_POINTS_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_RF_Points_HY2025"
TERRAIN_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_Terrain_GLO30"

# -------------------------------------------------------------------------
# Temporal-window choice
# -------------------------------------------------------------------------

# Recommended working choice after the preliminary comparison.
# Easy alternatives: "Mar_Aug", "May_Aug", "Full_HY"
TEMPORAL_WINDOW = "Apr_Aug"

WINDOW_TAGS = {
    "Mar_Aug": "MarAug",
    "Apr_Aug": "AprAug",
    "May_Aug": "MayAug",
    "Full_HY": "FullHY",
}

if TEMPORAL_WINDOW not in WINDOW_TAGS:
    raise ValueError(
        f"Unknown TEMPORAL_WINDOW={TEMPORAL_WINDOW}. "
        f"Choose from {list(WINDOW_TAGS)}"
    )

WINDOW_TAG = WINDOW_TAGS[TEMPORAL_WINDOW]

TRAINING_YEAR = 2025
HABITAT_YEARS = list(range(2018, 2026))

# -------------------------------------------------------------------------
# Local outputs
# -------------------------------------------------------------------------

LOCAL_ROOT = Path(
    r"C:\Users\PangY\OneDrive - Smithsonian Institution\Bustard"
    r"\01_Data\Habitat mapping\Annual_Classification\Sentinel2"
    rf"\{WINDOW_TAG}"
)

MANIFEST_DIR = LOCAL_ROOT / "Manifest"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

DRIVE_FOLDER = f"Eastern_Morocco_Habitat_S2_{WINDOW_TAG}_Annual"

# -------------------------------------------------------------------------
# Rebuild switches
#
# Keep all False for normal/resumed runs.
# -------------------------------------------------------------------------

REBUILD_SPECTRAL_ASSETS = False
REBUILD_TRAINING_SAMPLE_ASSET = False
RETRAIN_CLASSIFIER = False
OVERWRITE_MAP_ASSETS = False

# -------------------------------------------------------------------------
# Export switches
# -------------------------------------------------------------------------

EXPORT_MAPS_TO_ASSET = True
EXPORT_MAPS_TO_DRIVE = True

# valid_obs is already stored as a band in every annual spectral asset.
# Turn this on only if you also want local valid-observation GeoTIFFs.
EXPORT_VALID_OBS_TO_DRIVE = False

# If a map asset already existed before the current run, do not automatically
# create another Drive export (avoids duplicate files on reruns).
# Set True only when you intentionally need Drive copies of existing assets.
EXPORT_EXISTING_MAP_ASSETS_TO_DRIVE = False

# -------------------------------------------------------------------------
# Conservative task behavior
# -------------------------------------------------------------------------

POLL_SECONDS = 30
MAX_ACTIVE_DRIVE_TASKS = 2
WAIT_FOR_DRIVE_AT_END = False

# -------------------------------------------------------------------------
# Sentinel-2 collections / masking
# -------------------------------------------------------------------------

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CLOUD_SCORE_BAND = "cs_cdf"

CLEAR_THRESHOLD = 0.60
MAX_SCENE_CLOUD_PERCENT = 20

# -------------------------------------------------------------------------
# Predictors
# -------------------------------------------------------------------------

S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
INDEX_BANDS = ["EVI", "MSAVI", "NDWI", "BSI"]
SPECTRAL_BANDS = S2_BANDS + INDEX_BANDS

TERRAIN_BANDS = ["Elevation", "Slope", "Northness", "Eastness"]

PREDICTOR_BANDS = SPECTRAL_BANDS + TERRAIN_BANDS

VALID_OBS_BAND = "valid_obs"

# -------------------------------------------------------------------------
# Training fields
# -------------------------------------------------------------------------

CLASS_FIELD = "class_id"
FEATURE_ID_FIELD = "feature_id"
FOLD_FIELD = "cv_fold"
SAMPLE_ID_FIELD = "sample_id"

SAMPLE_PROPERTIES = [
    SAMPLE_ID_FIELD,
    FEATURE_ID_FIELD,
    CLASS_FIELD,
    FOLD_FIELD,
]

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

# -------------------------------------------------------------------------
# Production GEE Random Forest
# -------------------------------------------------------------------------

RANDOM_SEED = 42

RF_TREES = 500
GEE_VARIABLES_PER_SPLIT = 6
GEE_MIN_LEAF_POPULATION = 2
GEE_BAG_FRACTION = 0.632

# -------------------------------------------------------------------------
# Sampling / export grid
# -------------------------------------------------------------------------

SAMPLE_SCALE = 10
SAMPLE_TILE_SCALE = 16

EXPORT_CRS = "EPSG:32630"
EXPORT_TRANSFORM = [10, 0, 0, 0, -10, 10_000_000]

MAX_PIXELS = 1e13
DRIVE_NODATA = 255

# Optional map masking based on the annual number of valid S2 observations.
# Keep False until you decide on a defensible threshold.
MASK_LOW_OBSERVATION_PIXELS = False
MIN_VALID_OBS = 5


# =============================================================================
# 1. LOGGING / TASK HELPERS
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

LOGGER = logging.getLogger("s2_annual_habitat_production")

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "CANCEL_REQUESTED",
}

ACTIVE_STATES = {
    "READY",
    "RUNNING",
}


def initialize_ee() -> None:
    """Initialize Earth Engine, authenticating only if needed."""
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        LOGGER.info("Earth Engine authentication required.")
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)

    LOGGER.info(
        "Earth Engine initialized with project '%s'.",
        EE_PROJECT,
    )


def asset_exists(asset_id: str) -> bool:
    """Return True if an Earth Engine asset is accessible."""
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def delete_asset(asset_id: str) -> None:
    """Delete an existing Earth Engine asset."""
    if asset_exists(asset_id):
        LOGGER.warning(
            "Deleting Earth Engine asset: %s",
            asset_id,
        )
        ee.data.deleteAsset(asset_id)


def wait_for_task(
    task: ee.batch.Task,
    label: str,
) -> Dict[str, Any]:
    """Wait for a batch task and raise a useful error on failure."""
    previous_state: Optional[str] = None

    while True:
        status = task.status()
        state = str(status.get("state", "UNKNOWN"))

        if state != previous_state:
            LOGGER.info(
                "Task %-50s %s",
                label,
                state,
            )
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


def throttle_drive_tasks(
    tasks: Dict[str, ee.batch.Task],
) -> None:
    """Keep only a small number of this script's Drive tasks active."""
    if MAX_ACTIVE_DRIVE_TASKS <= 0:
        return

    while True:
        active = [
            name
            for name, task in tasks.items()
            if str(
                task.status().get("state", "UNKNOWN")
            ) in ACTIVE_STATES
        ]

        if len(active) < MAX_ACTIVE_DRIVE_TASKS:
            return

        LOGGER.info(
            "Drive export throttle: %d task(s) active; waiting.",
            len(active),
        )

        time.sleep(POLL_SECONDS)


def wait_for_drive_tasks(
    tasks: Dict[str, ee.batch.Task],
) -> None:
    """Optionally wait for all Drive exports started in the current run."""
    for name, task in tasks.items():
        state = str(
            task.status().get("state", "UNKNOWN")
        )

        if state not in TERMINAL_STATES:
            wait_for_task(task, name)


# =============================================================================
# 2. TEMPORAL WINDOW / ASSET NAMING
# =============================================================================

def window_dates(year: int) -> Tuple[str, str]:
    """
    Return start date and exclusive end date for the selected temporal window.
    """
    if TEMPORAL_WINDOW == "Mar_Aug":
        return f"{year}-03-01", f"{year}-09-01"

    if TEMPORAL_WINDOW == "Apr_Aug":
        return f"{year}-04-01", f"{year}-09-01"

    if TEMPORAL_WINDOW == "May_Aug":
        return f"{year}-05-01", f"{year}-09-01"

    if TEMPORAL_WINDOW == "Full_HY":
        return f"{year - 1}-09-01", f"{year}-09-01"

    raise ValueError(
        f"Unsupported temporal window: {TEMPORAL_WINDOW}"
    )


def spectral_asset_id(year: int) -> str:
    """
    Annual materialized S2 asset.

    Contains:
      B2 B3 B4 B8 B11 B12
      EVI MSAVI NDWI BSI
      valid_obs

    Terrain is NOT duplicated in annual assets; the one static terrain asset
    is reused when sampling/training/classifying.
    """
    return (
        f"{ASSET_ROOT}/"
        f"Eastern_Morocco_S2_Spectral_{WINDOW_TAG}_HY{year}"
    )


def training_sample_asset_id() -> str:
    """Materialized 17,175-row HY2025 production sample table."""
    return (
        f"{ASSET_ROOT}/"
        f"Eastern_Morocco_S2_RF_Samples_"
        f"{WINDOW_TAG}_HY{TRAINING_YEAR}_Fixed5Pts"
    )


def classifier_asset_id() -> str:
    """Production GEE RF classifier for the selected temporal window."""
    return (
        f"{ASSET_ROOT}/"
        f"Eastern_Morocco_S2_RF_Classifier_"
        f"{WINDOW_TAG}_HY{TRAINING_YEAR}_T{RF_TREES}"
    )


def habitat_map_asset_id(year: int) -> str:
    """Annual habitat-map asset."""
    return (
        f"{ASSET_ROOT}/"
        f"Eastern_Morocco_Habitat_S2_RF_"
        f"{WINDOW_TAG}_HY{year}_T{RF_TREES}"
    )


# =============================================================================
# 3. PREFLIGHT QA
# =============================================================================

def preflight_assets() -> None:
    """Check only the authoritative upstream assets needed by this script."""
    required = [
        ROI_ASSET,
        RF_POINTS_ASSET,
        TERRAIN_ASSET,
    ]

    missing = [
        asset_id
        for asset_id in required
        if not asset_exists(asset_id)
    ]

    if missing:
        raise FileNotFoundError(
            "Required Earth Engine asset(s) are missing:\n  "
            + "\n  ".join(missing)
        )

    points = ee.FeatureCollection(
        RF_POINTS_ASSET
    )

    n_points = int(
        points.size().getInfo()
    )

    n_polygons = int(
        points.aggregate_count_distinct(
            FEATURE_ID_FIELD
        ).getInfo()
    )

    properties = (
        ee.Feature(points.first())
        .propertyNames()
        .getInfo()
    )

    missing_properties = [
        prop
        for prop in SAMPLE_PROPERTIES
        if prop not in properties
    ]

    if missing_properties:
        raise KeyError(
            "RF point asset is missing required fields: "
            f"{missing_properties}"
        )

    LOGGER.info(
        "Existing RF points: %d",
        n_points,
    )

    LOGGER.info(
        "Existing source polygons: %d",
        n_polygons,
    )

    if n_points != 17_175:
        LOGGER.warning(
            "Expected 17,175 RF points, but found %d.",
            n_points,
        )

    if n_polygons != 3_435:
        LOGGER.warning(
            "Expected 3,435 source polygons, but found %d.",
            n_polygons,
        )

    terrain_band_names = (
        ee.Image(TERRAIN_ASSET)
        .bandNames()
        .getInfo()
    )

    missing_terrain = [
        band
        for band in TERRAIN_BANDS
        if band not in terrain_band_names
    ]

    if missing_terrain:
        raise KeyError(
            "Terrain asset is missing bands: "
            f"{missing_terrain}"
        )

    LOGGER.info(
        "Authoritative upstream assets passed preflight QA."
    )


# =============================================================================
# 4. SENTINEL-2 PREPROCESSING
# =============================================================================

def prepare_s2(
    image: ee.Image,
) -> ee.Image:
    """Scale S2 SR and apply Cloud Score+ and SCL masks."""
    image = ee.Image(image)

    clear_score = (
        image.select(CLOUD_SCORE_BAND)
        .gte(CLEAR_THRESHOLD)
    )

    scl = image.select("SCL")

    scl_clear = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )

    return (
        image.select(S2_BANDS)
        .multiply(0.0001)
        .updateMask(clear_score)
        .updateMask(scl_clear)
        .copyProperties(
            image,
            [
                "system:time_start",
                "system:index",
            ],
        )
    )


def add_indices(
    image: ee.Image,
) -> ee.Image:
    """Add EVI, MSAVI, NDWI, and BSI to a reflectance composite."""
    blue = image.select("B2")
    green = image.select("B3")
    red = image.select("B4")
    nir = image.select("B8")
    swir1 = image.select("B11")

    evi = (
        image.expression(
            (
                "2.5 * (NIR - RED) / "
                "(NIR + 6.0 * RED - "
                "7.5 * BLUE + 1.0)"
            ),
            {
                "NIR": nir,
                "RED": red,
                "BLUE": blue,
            },
        )
        .rename("EVI")
    )

    msavi_term = (
        nir.multiply(2)
        .add(1)
    )

    msavi_disc = (
        msavi_term.pow(2)
        .subtract(
            nir.subtract(red).multiply(8)
        )
        .max(0)
    )

    msavi = (
        msavi_term
        .subtract(msavi_disc.sqrt())
        .divide(2)
        .rename("MSAVI")
    )

    ndwi = (
        image.expression(
            "(GREEN - NIR) / (GREEN + NIR)",
            {
                "GREEN": green,
                "NIR": nir,
            },
        )
        .rename("NDWI")
    )

    bsi = (
        image.expression(
            (
                "((SWIR + RED) - (NIR + BLUE)) / "
                "((SWIR + RED) + (NIR + BLUE))"
            ),
            {
                "SWIR": swir1,
                "RED": red,
                "NIR": nir,
                "BLUE": blue,
            },
        )
        .rename("BSI")
    )

    return image.addBands(
        [
            evi,
            msavi,
            ndwi,
            bsi,
        ]
    )


def get_s2_collection(
    year: int,
    roi: ee.Geometry,
) -> ee.ImageCollection:
    """Return the masked S2 collection for one annual mapping window."""
    start_date, end_date = window_dates(year)

    s2 = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(roi)
        .filterDate(
            start_date,
            end_date,
        )
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                MAX_SCENE_CLOUD_PERCENT,
            )
        )
    )

    cloud_score = (
        ee.ImageCollection(
            CLOUD_SCORE_COLLECTION
        )
        .filterBounds(roi)
        .filterDate(
            start_date,
            end_date,
        )
    )

    return (
        s2.linkCollection(
            cloud_score,
            [CLOUD_SCORE_BAND],
        )
        .map(prepare_s2)
    )


def build_annual_spectral_image(
    year: int,
    roi: ee.Geometry,
) -> ee.Image:
    """
    Build the annual S2 spectral/index image.

    Heavy Sentinel-2 computation happens here, but only when the annual
    spectral asset does not already exist.

    Output bands:
      10 spectral/index predictors + valid_obs
    """
    start_date, end_date = window_dates(year)

    clean = get_s2_collection(
        year,
        roi,
    )

    # Median reflectance FIRST, then calculate indices.
    # This definition is kept identical for every year.
    composite = (
        clean.select(S2_BANDS)
        .median()
    )

    spectral = (
        add_indices(composite)
        .select(SPECTRAL_BANDS)
        .float()
    )

    valid_obs = (
        clean.select("B2")
        .count()
        .rename(VALID_OBS_BAND)
        .unmask(0)
        .toFloat()
    )

    return (
        spectral
        .addBands(valid_obs)
        .clip(roi)
        .set(
            {
                "sensor": "Sentinel-2",
                "habitat_year": year,
                "temporal_window": TEMPORAL_WINDOW,
                "window_tag": WINDOW_TAG,
                "start_date": start_date,
                "end_date_exclusive": end_date,
                "cloud_score_band": CLOUD_SCORE_BAND,
                "clear_threshold": CLEAR_THRESHOLD,
                "max_scene_cloud_percent":
                    MAX_SCENE_CLOUD_PERCENT,
                "composite": "median_reflectance_then_indices",
            }
        )
    )


# =============================================================================
# 5. MATERIALIZE / REUSE ANNUAL SPECTRAL ASSETS
# =============================================================================

def ensure_spectral_asset(
    year: int,
    roi: ee.Geometry,
) -> Tuple[ee.Image, str, Optional[int]]:
    """
    Ensure the annual S2 spectral/index image is stored as an EE asset.

    This is the main computation-cache step. Once created, all later
    sampling/classification uses the stored asset instead of rebuilding
    Sentinel-2 composites.
    """
    asset_id = spectral_asset_id(year)

    if (
        asset_exists(asset_id)
        and not REBUILD_SPECTRAL_ASSETS
    ):
        LOGGER.info(
            "Reusing annual S2 spectral asset: %s",
            asset_id,
        )

        return (
            ee.Image(asset_id),
            "existing",
            None,
        )

    if asset_exists(asset_id):
        delete_asset(asset_id)

    clean = get_s2_collection(
        year,
        roi,
    )

    # Small metadata request only; does not compute the raster composite.
    scene_count = int(
        clean.size().getInfo()
    )

    if scene_count == 0:
        raise RuntimeError(
            f"HY{year} has no Sentinel-2 scenes for "
            f"{TEMPORAL_WINDOW}."
        )

    LOGGER.info(
        "HY%d | %s | filtered S2 granules: %d",
        year,
        TEMPORAL_WINDOW,
        scene_count,
    )

    image = build_annual_spectral_image(
        year,
        roi,
    ).set(
        "scene_granules",
        scene_count,
    )

    task = ee.batch.Export.image.toAsset(
        image=image,
        description=(
            f"Eastern_Morocco_S2_Spectral_"
            f"{WINDOW_TAG}_HY{year}"
        ),
        assetId=asset_id,
        region=roi,
        crs=EXPORT_CRS,
        crsTransform=EXPORT_TRANSFORM,
        maxPixels=MAX_PIXELS,
        pyramidingPolicy={
            ".default": "mean",
        },
    )

    task.start()

    LOGGER.info(
        "Started HY%d spectral-asset export: %s",
        year,
        task.id,
    )

    wait_for_task(
        task,
        f"S2 spectral {WINDOW_TAG} HY{year}",
    )

    return (
        ee.Image(asset_id),
        "completed",
        scene_count,
    )


# =============================================================================
# 6. BUILD FULL PREDICTOR IMAGE FROM REUSED ASSETS
# =============================================================================

def predictor_from_assets(
    spectral_image: ee.Image,
    terrain: ee.Image,
) -> ee.Image:
    """
    Combine an annual stored S2 asset with the one reused terrain asset.

    Terrain is intentionally not duplicated into every annual S2 asset.
    """
    return (
        spectral_image
        .select(SPECTRAL_BANDS)
        .addBands(terrain)
        .select(PREDICTOR_BANDS)
        .float()
    )


# =============================================================================
# 7. MATERIALIZE / REUSE HY2025 17,175-POINT SAMPLE TABLE
# =============================================================================

def ensure_training_sample_asset(
    training_spectral: ee.Image,
    terrain: ee.Image,
) -> ee.FeatureCollection:
    """
    Extract production predictors at the existing 17,175 fixed points ONCE.

    Because the HY2025 S2 image and terrain are already materialized assets,
    this is much lighter than sampling a live Sentinel-2 computation graph.
    """
    asset_id = training_sample_asset_id()

    if (
        asset_exists(asset_id)
        and not REBUILD_TRAINING_SAMPLE_ASSET
    ):
        LOGGER.info(
            "Reusing HY2025 training sample asset: %s",
            asset_id,
        )

        return ee.FeatureCollection(
            asset_id
        )

    if asset_exists(asset_id):
        delete_asset(asset_id)

    points = ee.FeatureCollection(
        RF_POINTS_ASSET
    )

    predictor = predictor_from_assets(
        training_spectral,
        terrain,
    )

    # Include valid_obs for later QA, but it is not used as an RF predictor.
    sample_image = (
        predictor
        .addBands(
            training_spectral.select(
                VALID_OBS_BAND
            )
        )
    )

    samples = (
        sample_image
        .sampleRegions(
            collection=points,
            properties=SAMPLE_PROPERTIES,
            scale=SAMPLE_SCALE,
            geometries=True,
            tileScale=SAMPLE_TILE_SCALE,
        )
        .filter(
            ee.Filter.notNull(
                PREDICTOR_BANDS
                + [
                    CLASS_FIELD,
                    FEATURE_ID_FIELD,
                    FOLD_FIELD,
                    SAMPLE_ID_FIELD,
                ]
            )
        )
    )

    task = ee.batch.Export.table.toAsset(
        collection=samples,
        description=(
            f"Eastern_Morocco_S2_RF_Samples_"
            f"{WINDOW_TAG}_HY{TRAINING_YEAR}_Fixed5Pts"
        ),
        assetId=asset_id,
    )

    task.start()

    LOGGER.info(
        "Started HY%d production sample-table export: %s",
        TRAINING_YEAR,
        task.id,
    )

    wait_for_task(
        task,
        f"HY{TRAINING_YEAR} {WINDOW_TAG} training samples",
    )

    return ee.FeatureCollection(
        asset_id
    )


# =============================================================================
# 8. TRAIN / REUSE THE PRODUCTION GEE CLASSIFIER
# =============================================================================

def build_gee_classifier(
    samples: ee.FeatureCollection,
) -> ee.Classifier:
    """Train the deployable unweighted Features6_Leaf2 Smile RF."""
    clean = samples.filter(
        ee.Filter.notNull(
            PREDICTOR_BANDS
            + [
                CLASS_FIELD,
            ]
        )
    )

    return (
        ee.Classifier.smileRandomForest(
            numberOfTrees=RF_TREES,
            variablesPerSplit=
                GEE_VARIABLES_PER_SPLIT,
            minLeafPopulation=
                GEE_MIN_LEAF_POPULATION,
            bagFraction=
                GEE_BAG_FRACTION,
            seed=
                RANDOM_SEED,
        )
        .train(
            features=clean,
            classProperty=CLASS_FIELD,
            inputProperties=PREDICTOR_BANDS,
        )
    )


def ensure_classifier_asset(
    samples: ee.FeatureCollection,
) -> ee.Classifier:
    """Train once, save once, then load the classifier on future runs."""
    asset_id = classifier_asset_id()

    if (
        asset_exists(asset_id)
        and not RETRAIN_CLASSIFIER
    ):
        LOGGER.info(
            "Reusing classifier asset: %s",
            asset_id,
        )

        return ee.Classifier.load(
            asset_id
        )

    if asset_exists(asset_id):
        delete_asset(asset_id)

    classifier = build_gee_classifier(
        samples
    )

    task = ee.batch.Export.classifier.toAsset(
        classifier=classifier,
        description=(
            f"Eastern_Morocco_S2_RF_Classifier_"
            f"{WINDOW_TAG}_HY{TRAINING_YEAR}_T{RF_TREES}"
        ),
        assetId=asset_id,
    )

    task.start()

    LOGGER.info(
        "Started production classifier export: %s",
        task.id,
    )

    wait_for_task(
        task,
        f"{WINDOW_TAG} HY{TRAINING_YEAR} classifier",
    )

    return ee.Classifier.load(
        asset_id
    )


# =============================================================================
# 9. CLASSIFY ONE YEAR FROM MATERIALIZED ASSETS
# =============================================================================

def classify_year(
    year: int,
    spectral_image: ee.Image,
    terrain: ee.Image,
    classifier: ee.Classifier,
    roi: ee.Geometry,
) -> ee.Image:
    """Classify one year from already-materialized predictor inputs."""
    start_date, end_date = window_dates(
        year
    )

    predictor = predictor_from_assets(
        spectral_image,
        terrain,
    )

    classified = (
        predictor
        .classify(classifier)
        .rename("class_id")
        .toUint8()
    )

    if MASK_LOW_OBSERVATION_PIXELS:
        classified = classified.updateMask(
            spectral_image
            .select(VALID_OBS_BAND)
            .gte(MIN_VALID_OBS)
        )

    return (
        classified
        .clip(roi)
        .set(
            {
                "sensor": "Sentinel-2",
                "habitat_year": year,
                "temporal_window": TEMPORAL_WINDOW,
                "window_tag": WINDOW_TAG,
                "start_date": start_date,
                "end_date_exclusive": end_date,
                "training_year": TRAINING_YEAR,
                "training_sample_asset":
                    training_sample_asset_id(),
                "classifier_asset":
                    classifier_asset_id(),
                "spectral_asset":
                    spectral_asset_id(year),
                "terrain_asset":
                    TERRAIN_ASSET,
                "trees": RF_TREES,
                "variables_per_split":
                    GEE_VARIABLES_PER_SPLIT,
                "min_leaf_population":
                    GEE_MIN_LEAF_POPULATION,
                "bag_fraction":
                    GEE_BAG_FRACTION,
                "low_obs_mask_applied":
                    MASK_LOW_OBSERVATION_PIXELS,
                "min_valid_obs":
                    MIN_VALID_OBS
                    if MASK_LOW_OBSERVATION_PIXELS
                    else -1,
            }
        )
    )


# =============================================================================
# 10. MATERIALIZE / REUSE ANNUAL HABITAT MAPS
# =============================================================================

def ensure_map_asset(
    year: int,
    classified: ee.Image,
    roi: ee.Geometry,
) -> Tuple[ee.Image, str]:
    """Store each annual habitat map once, then reuse it on reruns."""
    asset_id = habitat_map_asset_id(
        year
    )

    if not EXPORT_MAPS_TO_ASSET:
        return (
            classified,
            "not_requested",
        )

    if (
        asset_exists(asset_id)
        and not OVERWRITE_MAP_ASSETS
    ):
        LOGGER.info(
            "Reusing annual habitat-map asset: %s",
            asset_id,
        )

        return (
            ee.Image(asset_id),
            "existing",
        )

    if asset_exists(asset_id):
        delete_asset(asset_id)

    task = ee.batch.Export.image.toAsset(
        image=classified,
        description=(
            f"Eastern_Morocco_Habitat_S2_RF_"
            f"{WINDOW_TAG}_HY{year}_T{RF_TREES}"
        ),
        assetId=asset_id,
        region=roi,
        crs=EXPORT_CRS,
        crsTransform=EXPORT_TRANSFORM,
        maxPixels=MAX_PIXELS,
        pyramidingPolicy={
            ".default": "mode",
        },
    )

    task.start()

    LOGGER.info(
        "Started HY%d habitat-map asset export: %s",
        year,
        task.id,
    )

    wait_for_task(
        task,
        f"Habitat map {WINDOW_TAG} HY{year}",
    )

    return (
        ee.Image(asset_id),
        "completed",
    )


# =============================================================================
# 11. OPTIONAL GOOGLE DRIVE EXPORTS
# =============================================================================

def start_map_drive_export(
    year: int,
    map_image: ee.Image,
    roi: ee.Geometry,
    drive_tasks: Dict[str, ee.batch.Task],
) -> Optional[ee.batch.Task]:
    """Export an already-materialized habitat-map asset to Drive."""
    if not EXPORT_MAPS_TO_DRIVE:
        return None

    throttle_drive_tasks(
        drive_tasks
    )

    name = (
        f"Eastern_Morocco_Habitat_S2_RF_"
        f"{WINDOW_TAG}_HY{year}_T{RF_TREES}"
    )

    drive_image = (
        map_image
        .unmask(
            value=DRIVE_NODATA,
            sameFootprint=False,
        )
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
        formatOptions={
            "cloudOptimized": True,
            "noData": DRIVE_NODATA,
        },
        skipEmptyTiles=True,
    )

    task.start()

    drive_tasks[name] = task

    LOGGER.info(
        "Started HY%d map Drive export: %s",
        year,
        task.id,
    )

    return task


def start_valid_obs_drive_export(
    year: int,
    spectral_image: ee.Image,
    roi: ee.Geometry,
    drive_tasks: Dict[str, ee.batch.Task],
) -> Optional[ee.batch.Task]:
    """Optionally export valid_obs from the stored annual S2 asset."""
    if not EXPORT_VALID_OBS_TO_DRIVE:
        return None

    throttle_drive_tasks(
        drive_tasks
    )

    name = (
        f"Eastern_Morocco_S2_ValidObs_"
        f"{WINDOW_TAG}_HY{year}"
    )

    image = (
        spectral_image
        .select(VALID_OBS_BAND)
        .clip(roi)
        .toUint16()
    )

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"{name}_Drive",
        folder=DRIVE_FOLDER,
        fileNamePrefix=name,
        region=roi,
        crs=EXPORT_CRS,
        crsTransform=EXPORT_TRANSFORM,
        maxPixels=MAX_PIXELS,
        fileFormat="GeoTIFF",
        formatOptions={
            "cloudOptimized": True,
        },
        skipEmptyTiles=True,
    )

    task.start()

    drive_tasks[name] = task

    LOGGER.info(
        "Started HY%d valid_obs Drive export: %s",
        year,
        task.id,
    )

    return task


# =============================================================================
# 12. ANNUAL PRODUCTION LOOP
# =============================================================================

def run_annual_maps(
    classifier: ee.Classifier,
    roi: ee.Geometry,
    terrain: ee.Image,
) -> pd.DataFrame:
    """
    Sequential annual workflow:

      ensure/reuse stored S2 spectral asset
        -> classify from stored assets
        -> ensure/reuse habitat-map asset
        -> optional Drive export from stored map asset

    Only one heavy annual S2 asset computation is run at a time.
    """
    drive_tasks: Dict[
        str,
        ee.batch.Task,
    ] = {}

    rows: List[
        Dict[str, Any]
    ] = []

    for year in HABITAT_YEARS:
        LOGGER.info(
            "=" * 80
        )

        LOGGER.info(
            "ANNUAL MAP | HY%d | %s",
            year,
            TEMPORAL_WINDOW,
        )

        start_date, end_date = window_dates(
            year
        )

        # -------------------------------------------------------------
        # A. Materialize/reuse annual S2 spectral/index asset.
        # -------------------------------------------------------------

        spectral_image, spectral_status, scene_count = (
            ensure_spectral_asset(
                year,
                roi,
            )
        )

        # -------------------------------------------------------------
        # B. Classification graph uses only STORED S2 + STORED terrain
        #    + STORED classifier.
        # -------------------------------------------------------------

        classified = classify_year(
            year,
            spectral_image,
            terrain,
            classifier,
            roi,
        )

        map_image, map_status = (
            ensure_map_asset(
                year,
                classified,
                roi,
            )
        )

        # -------------------------------------------------------------
        # C. Drive export only from materialized map asset.
        #
        # If the asset existed before this run, skip the Drive export
        # by default to avoid accidental duplicate files.
        # -------------------------------------------------------------

        should_export_existing = (
            EXPORT_EXISTING_MAP_ASSETS_TO_DRIVE
        )

        should_start_drive = (
            EXPORT_MAPS_TO_DRIVE
            and (
                map_status == "completed"
                or should_export_existing
            )
        )

        if should_start_drive:
            map_drive_task = (
                start_map_drive_export(
                    year,
                    map_image,
                    roi,
                    drive_tasks,
                )
            )
        else:
            map_drive_task = None

            if (
                EXPORT_MAPS_TO_DRIVE
                and map_status == "existing"
            ):
                LOGGER.info(
                    "HY%d map asset already existed; "
                    "Drive export skipped to avoid duplicates. "
                    "Set EXPORT_EXISTING_MAP_ASSETS_TO_DRIVE=True "
                    "if a Drive copy is needed.",
                    year,
                )

        valid_obs_drive_task = (
            start_valid_obs_drive_export(
                year,
                spectral_image,
                roi,
                drive_tasks,
            )
        )

        row = {
            "sensor": "Sentinel-2",
            "habitat_year": year,
            "temporal_window":
                TEMPORAL_WINDOW,
            "start_date":
                start_date,
            "end_date_exclusive":
                end_date,
            "scene_count_if_created":
                scene_count,
            "spectral_asset":
                spectral_asset_id(year),
            "spectral_asset_status":
                spectral_status,
            "habitat_map_asset":
                habitat_map_asset_id(year),
            "habitat_map_asset_status":
                map_status,
            "classifier_asset":
                classifier_asset_id(),
            "training_sample_asset":
                training_sample_asset_id(),
            "drive_folder":
                DRIVE_FOLDER
                if EXPORT_MAPS_TO_DRIVE
                else "",
            "map_drive_task_id":
                map_drive_task.id
                if map_drive_task is not None
                else "",
            "valid_obs_drive_task_id":
                valid_obs_drive_task.id
                if valid_obs_drive_task is not None
                else "",
        }

        rows.append(
            row
        )

        pd.DataFrame(
            rows
        ).to_csv(
            MANIFEST_DIR
            / "S2_Annual_Export_Manifest.csv",
            index=False,
        )

    if WAIT_FOR_DRIVE_AT_END:
        wait_for_drive_tasks(
            drive_tasks
        )

    manifest = pd.DataFrame(
        rows
    )

    if drive_tasks:
        task_states = {
            name: str(
                task.status().get(
                    "state",
                    "UNKNOWN",
                )
            )
            for name, task
            in drive_tasks.items()
        }

        if not manifest.empty:
            manifest[
                "map_drive_state_at_script_end"
            ] = manifest[
                "habitat_year"
            ].map(
                lambda year: task_states.get(
                    (
                        f"Eastern_Morocco_Habitat_S2_RF_"
                        f"{WINDOW_TAG}_HY{int(year)}_T{RF_TREES}"
                    ),
                    "",
                )
            )

    manifest.to_csv(
        MANIFEST_DIR
        / "S2_Annual_Export_Manifest.csv",
        index=False,
    )

    return manifest


# =============================================================================
# 13. RUN CONFIGURATION RECORD
# =============================================================================

def save_run_configuration() -> None:
    """Write a local JSON record of the production settings."""
    config = {
        "ee_project":
            EE_PROJECT,
        "asset_root":
            ASSET_ROOT,
        "roi_asset":
            ROI_ASSET,
        "rf_points_asset":
            RF_POINTS_ASSET,
        "terrain_asset":
            TERRAIN_ASSET,
        "temporal_window":
            TEMPORAL_WINDOW,
        "window_tag":
            WINDOW_TAG,
        "training_year":
            TRAINING_YEAR,
        "habitat_years":
            HABITAT_YEARS,
        "spectral_bands":
            SPECTRAL_BANDS,
        "terrain_bands":
            TERRAIN_BANDS,
        "predictor_bands":
            PREDICTOR_BANDS,
        "valid_obs_band":
            VALID_OBS_BAND,
        "cloud_score_threshold":
            CLEAR_THRESHOLD,
        "max_scene_cloud_percent":
            MAX_SCENE_CLOUD_PERCENT,
        "rf_trees":
            RF_TREES,
        "variables_per_split":
            GEE_VARIABLES_PER_SPLIT,
        "min_leaf_population":
            GEE_MIN_LEAF_POPULATION,
        "bag_fraction":
            GEE_BAG_FRACTION,
        "random_seed":
            RANDOM_SEED,
        "export_crs":
            EXPORT_CRS,
        "export_transform":
            EXPORT_TRANSFORM,
        "training_sample_asset":
            training_sample_asset_id(),
        "classifier_asset":
            classifier_asset_id(),
        "drive_folder":
            DRIVE_FOLDER,
        "mask_low_observation_pixels":
            MASK_LOW_OBSERVATION_PIXELS,
        "min_valid_obs":
            MIN_VALID_OBS,
        "class_map":
            CLASS_MAP,
    }

    path = (
        MANIFEST_DIR
        / "S2_Run_Configuration.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            config,
            fp,
            indent=2,
        )

    LOGGER.info(
        "Saved run configuration: %s",
        path,
    )


# =============================================================================
# 14. MAIN
# =============================================================================

def main() -> None:
    initialize_ee()

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_run_configuration()
    preflight_assets()

    roi = (
        ee.FeatureCollection(
            ROI_ASSET
        )
        .geometry()
    )

    # Static terrain is loaded once and reused everywhere.
    terrain = (
        ee.Image(
            TERRAIN_ASSET
        )
        .select(
            TERRAIN_BANDS
        )
        .resample(
            "bilinear"
        )
    )

    LOGGER.info(
        "=" * 80
    )

    LOGGER.info(
        "TRAINING-YEAR SETUP | HY%d | %s",
        TRAINING_YEAR,
        TEMPORAL_WINDOW,
    )

    # ---------------------------------------------------------------------
    # 1. Materialize/reuse HY2025 S2 spectral asset.
    # ---------------------------------------------------------------------

    training_spectral, training_spectral_status, _ = (
        ensure_spectral_asset(
            TRAINING_YEAR,
            roi,
        )
    )

    LOGGER.info(
        "HY%d spectral asset status: %s",
        TRAINING_YEAR,
        training_spectral_status,
    )

    # ---------------------------------------------------------------------
    # 2. Materialize/reuse 17,175-point HY2025 training table.
    # ---------------------------------------------------------------------

    training_samples = (
        ensure_training_sample_asset(
            training_spectral,
            terrain,
        )
    )

    # ---------------------------------------------------------------------
    # 3. Train/reuse the selected-window production classifier.
    # ---------------------------------------------------------------------

    classifier = (
        ensure_classifier_asset(
            training_samples
        )
    )

    LOGGER.info(
        "Classifier ready: %s",
        classifier_asset_id(),
    )

    # ---------------------------------------------------------------------
    # 4. Produce/reuse HY2018-HY2025 maps.
    #
    # HY2025 immediately reuses the spectral asset built above.
    # ---------------------------------------------------------------------

    manifest = run_annual_maps(
        classifier,
        roi,
        terrain,
    )

    LOGGER.info(
        "=" * 80
    )

    LOGGER.info(
        "Sentinel-2 annual habitat production finished."
    )

    LOGGER.info(
        "Temporal window: %s",
        TEMPORAL_WINDOW,
    )

    LOGGER.info(
        "Training sample asset: %s",
        training_sample_asset_id(),
    )

    LOGGER.info(
        "Classifier asset: %s",
        classifier_asset_id(),
    )

    LOGGER.info(
        "Manifest: %s",
        MANIFEST_DIR
        / "S2_Annual_Export_Manifest.csv",
    )

    LOGGER.info(
        "Drive folder: %s",
        DRIVE_FOLDER,
    )

    if (
        EXPORT_MAPS_TO_DRIVE
        and not WAIT_FOR_DRIVE_AT_END
    ):
        LOGGER.info(
            "Drive tasks may continue in Earth Engine "
            "after the Python script exits."
        )

    if not manifest.empty:
        LOGGER.info(
            "\n%s",
            manifest[
                [
                    "habitat_year",
                    "spectral_asset_status",
                    "habitat_map_asset_status",
                ]
            ]
            .to_string(
                index=False
            ),
        )


if __name__ == "__main__":
    main()
