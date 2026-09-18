#!/usr/bin/env python
"""Sentinel-2 annual habitat classification, Eastern Morocco.

This is the cleaned production script for the agreed HY2025 Random Forest
workflow and the annual Sentinel-2 series.

Key design choices
------------------
- Habitat year: 1 September (year-1) through 31 August (year).
- Annual Sentinel-2 series: HY2018-HY2025.
- HY2025 training predictor stack is REUSED from the existing Earth Engine
  asset to preserve the exact reference-year predictor definition.
- Fixed five-points-per-polygon reference locations are REUSED from the
  existing Earth Engine point asset. The discarded raster-wide stratified
  sample asset is never used.
- Local five-fold CV is grouped by source polygon through the preassigned
  ``cv_fold`` field.
- The operational GEE model is the deployable, unweighted Features6_Leaf2
  configuration: 500 trees, 6 variables per split, minimum leaf population 2,
  bag fraction 0.632, seed 42.
- The balanced-subsample model is retained only as a local sensitivity model
  because Earth Engine Smile Random Forest does not expose that class-weight
  option.
- Expensive GEE work is materialized with batch exports. No 500-tree
  ``confusionMatrix().getInfo()`` call is used.
- Annual maps are exported to Earth Engine Assets first and then to Google
  Drive from the stored map asset. This keeps the map export graph compact and
  makes reruns resumable.

Accuracy interpretation
-----------------------
The reportable accuracy is the polygon-grouped HY2025 five-fold CV. GEE
training/resubstitution and out-of-bag values are internal diagnostics only.
Historical annual maps are temporal transfers of the HY2025 classifier.
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

# Existing, authoritative assets developed in the HY2025 notebook workflow.
ROI_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_Working_Area"
RF_POINTS_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_RF_Points_HY2025"
TERRAIN_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_Terrain_GLO30"
S2_HY2025_SPECTRAL_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_S2_Spectral_HY2025"
S2_HY2025_PREDICTOR_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_PredictorStack_HY2025"

# Safe materialized products created/reused by this production script.
# These are NOT the discarded raster-wide stratified-sample product.
S2_FIXED_SAMPLE_ASSET = (
    f"{ASSET_ROOT}/Eastern_Morocco_S2_RF_Samples_Fixed5Pts_HY2025"
)
S2_CLASSIFIER_ASSET = (
    f"{ASSET_ROOT}/Eastern_Morocco_S2_RF_Classifier_Fixed5Pts_HY2025_T500"
)
S2_TRAINING_PREDICTIONS_ASSET = (
    f"{ASSET_ROOT}/Eastern_Morocco_S2_RF_TrainingPredictions_Fixed5Pts_HY2025_T500"
)

TRAINING_HABITAT_YEAR = 2025
HABITAT_YEARS = list(range(2018, 2026))

# Local outputs. The user's existing Smithsonian OneDrive hierarchy is reused.
LOCAL_ROOT = Path(
    r"C:\Users\PangY\OneDrive - Smithsonian Institution\Bustard"
    r"\01_Data\Habitat mapping\Annual_Classification\Sentinel2"
)
ACCURACY_DIR = LOCAL_ROOT / "Accuracy"
MANIFEST_DIR = LOCAL_ROOT / "Manifest"
RF_SAMPLE_DIR = Path(
    r"C:\Users\PangY\OneDrive - Smithsonian Institution\Bustard"
    r"\01_Data\Habitat mapping\RF_Sampling"
)
LOCAL_SAMPLE_CSV = RF_SAMPLE_DIR / "Eastern_Morocco_S2_RF_Samples_HY2025_Fixed5Pts.csv"
LOCAL_SAMPLE_XLSX = RF_SAMPLE_DIR / "Eastern_Morocco_S2_RF_Samples_HY2025_Fixed5Pts.xlsx"

# Google Drive folder used by Earth Engine export tasks.
DRIVE_FOLDER = "Eastern_Morocco_Habitat_S2_Annual"

# Rebuild switches. Keep False during normal reruns.
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

# Safe task behavior. Map assets are materialized one year at a time.
WAIT_FOR_CLASSIFIER = True
WAIT_FOR_MAP_ASSET = True
WAIT_FOR_DRIVE_AT_END = False
MAX_ACTIVE_DRIVE_TASKS = 2
POLL_SECONDS = 30

# Sentinel-2 collections and masking.
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CLOUD_SCORE_BAND = "cs_cdf"
CLEAR_THRESHOLD = 0.60
MAX_SCENE_CLOUD_PERCENT = 20

# Predictor definitions fixed from the HY2025 development workflow.
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
INDEX_BANDS = ["EVI", "MSAVI", "NDWI", "BSI"]
TERRAIN_BANDS = ["Elevation", "Slope", "Northness", "Eastness"]
PREDICTOR_BANDS = S2_BANDS + INDEX_BANDS + TERRAIN_BANDS

CLASS_FIELD = "class_id"
FEATURE_ID_FIELD = "feature_id"
FOLD_FIELD = "cv_fold"
SAMPLE_ID_FIELD = "sample_id"
SAMPLE_PROPERTIES = [SAMPLE_ID_FIELD, FEATURE_ID_FIELD, CLASS_FIELD, FOLD_FIELD]

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

# Final/deployable RF parameters.
RANDOM_SEED = 42
K_FOLDS = 5
RF_TREES = 500
GEE_VARIABLES_PER_SPLIT = 6
GEE_MIN_LEAF_POPULATION = 2
GEE_BAG_FRACTION = 0.632

# Fixed export grid. Using a fixed transform prevents annual grid shifts.
EXPORT_CRS = "EPSG:32630"
EXPORT_TRANSFORM = [10, 0, 0, 0, -10, 10_000_000]
SAMPLE_SCALE = 10
MAX_PIXELS = 1e13
DRIVE_NODATA = 255

# The six RF configurations already evaluated during tuning. Re-running them
# here reproduces the complete five-fold tuning table in the production run.
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
# 1. LOGGING AND BASIC EARTH ENGINE HELPERS
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("sentinel2_habitat")

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "CANCEL_REQUESTED"}
ACTIVE_STATES = {"READY", "RUNNING"}


def initialize_ee() -> None:
    """Initialize Earth Engine, authenticating only if required."""
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        LOGGER.info("Earth Engine authentication is required.")
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)
    LOGGER.info("Earth Engine initialized with project '%s'.", EE_PROJECT)


def asset_exists(asset_id: str) -> bool:
    """Return True when an Earth Engine asset is accessible."""
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
    """Wait for one batch task and raise a useful error if it fails."""
    previous_state: Optional[str] = None
    while True:
        status = task.status()
        state = str(status.get("state", "UNKNOWN"))
        if state != previous_state:
            LOGGER.info("Task %-45s %s", label, state)
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
    """Keep at most MAX_ACTIVE_DRIVE_TASKS of this script's Drive tasks active."""
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
        LOGGER.info(
            "Drive export throttle: %d task(s) active; waiting for a slot.",
            len(active),
        )
        time.sleep(POLL_SECONDS)


def wait_for_drive_tasks(tasks: Dict[str, ee.batch.Task]) -> None:
    for name, task in tasks.items():
        state = str(task.status().get("state", "UNKNOWN"))
        if state not in TERMINAL_STATES:
            wait_for_task(task, name)


def habitat_year_dates(year: int) -> Tuple[str, str]:
    return f"{year - 1}-09-01", f"{year}-09-01"


# =============================================================================
# 2. PREFLIGHT QA OF THE EXISTING HY2025 ASSETS
# =============================================================================


def preflight_assets() -> None:
    required = [
        ROI_ASSET,
        RF_POINTS_ASSET,
        TERRAIN_ASSET,
        S2_HY2025_PREDICTOR_ASSET,
    ]
    require_assets(required)

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

    predictor = ee.Image(S2_HY2025_PREDICTOR_ASSET)
    bands = predictor.bandNames().getInfo()
    missing_bands = [band for band in PREDICTOR_BANDS if band not in bands]
    if missing_bands:
        raise KeyError(
            f"Saved HY2025 predictor asset is missing bands: {missing_bands}"
        )

    terrain_bands = ee.Image(TERRAIN_ASSET).bandNames().getInfo()
    missing_terrain = [band for band in TERRAIN_BANDS if band not in terrain_bands]
    if missing_terrain:
        raise KeyError(f"Saved terrain asset is missing bands: {missing_terrain}")

    LOGGER.info("Existing HY2025 assets passed preflight QA.")


# =============================================================================
# 3. SENTINEL-2 ANNUAL PREDICTORS
# =============================================================================


def prepare_s2(image: ee.Image) -> ee.Image:
    image = ee.Image(image)
    clear_score = image.select(CLOUD_SCORE_BAND).gte(CLEAR_THRESHOLD)
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
        .copyProperties(image, ["system:time_start", "system:index"])
    )


def add_indices(image: ee.Image) -> ee.Image:
    blue = image.select("B2")
    green = image.select("B3")
    red = image.select("B4")
    nir = image.select("B8")
    swir1 = image.select("B11")

    evi = image.expression(
        "2.5 * (NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0)",
        {"NIR": nir, "RED": red, "BLUE": blue},
    ).rename("EVI")

    msavi_term = nir.multiply(2).add(1)
    msavi_disc = msavi_term.pow(2).subtract(nir.subtract(red).multiply(8)).max(0)
    msavi = msavi_term.subtract(msavi_disc.sqrt()).divide(2).rename("MSAVI")

    # Expression is used instead of normalizedDifference so the same formula is
    # portable to Landsat, where scaled SR values can occasionally be negative.
    ndwi = image.expression(
        "(GREEN - NIR) / (GREEN + NIR)",
        {"GREEN": green, "NIR": nir},
    ).rename("NDWI")

    bsi = image.expression(
        "((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))",
        {"SWIR": swir1, "RED": red, "NIR": nir, "BLUE": blue},
    ).rename("BSI")

    return image.addBands([evi, msavi, ndwi, bsi])


def get_s2_collection(year: int, roi: ee.Geometry) -> ee.ImageCollection:
    start_date, end_date = habitat_year_dates(year)
    s2 = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_SCENE_CLOUD_PERCENT))
    )
    cloud_score = (
        ee.ImageCollection(CLOUD_SCORE_COLLECTION)
        .filterBounds(roi)
        .filterDate(start_date, end_date)
    )
    return s2.linkCollection(cloud_score, [CLOUD_SCORE_BAND]).map(prepare_s2)


def build_s2_predictor_stack(
    year: int,
    roi: ee.Geometry,
    terrain: ee.Image,
) -> Tuple[ee.Image, ee.Image]:
    """Return the 14-band annual stack and valid-observation count."""
    if year == TRAINING_HABITAT_YEAR:
        # Critical for reproducibility: do not rebuild the training-year stack.
        stack = ee.Image(S2_HY2025_PREDICTOR_ASSET).select(PREDICTOR_BANDS).float()
        clean = get_s2_collection(year, roi)
        valid_obs = clean.select("B2").count().rename("valid_obs").toUint16()
        return stack, valid_obs

    clean = get_s2_collection(year, roi)
    composite = clean.median()
    spectral = add_indices(composite)
    valid_obs = clean.select("B2").count().rename("valid_obs").toUint16()
    stack = (
        spectral.select(S2_BANDS + INDEX_BANDS)
        .addBands(terrain)
        .select(PREDICTOR_BANDS)
        .float()
    )
    return stack, valid_obs


# =============================================================================
# 4. FIXED-POINT SAMPLE TABLE
# =============================================================================


def build_fixed_sample_fc() -> ee.FeatureCollection:
    """Extract HY2025 predictors at the 17,175 pre-defined reference points."""
    points = ee.FeatureCollection(RF_POINTS_ASSET)
    training_stack = ee.Image(S2_HY2025_PREDICTOR_ASSET).select(PREDICTOR_BANDS)
    samples = training_stack.sampleRegions(
        collection=points,
        properties=SAMPLE_PROPERTIES,
        scale=SAMPLE_SCALE,
        geometries=False,
        tileScale=4,
    )
    return samples.filter(
        ee.Filter.notNull(PREDICTOR_BANDS + [CLASS_FIELD, FEATURE_ID_FIELD, FOLD_FIELD])
    )


def ensure_fixed_sample_asset() -> ee.FeatureCollection:
    """Materialize/reuse the lightweight fixed-point predictor table in EE."""
    if asset_exists(S2_FIXED_SAMPLE_ASSET) and not REBUILD_FIXED_SAMPLE_ASSET:
        LOGGER.info("Reusing fixed-point sample asset: %s", S2_FIXED_SAMPLE_ASSET)
        return ee.FeatureCollection(S2_FIXED_SAMPLE_ASSET)

    if asset_exists(S2_FIXED_SAMPLE_ASSET):
        delete_asset(S2_FIXED_SAMPLE_ASSET)

    samples = build_fixed_sample_fc()
    task = ee.batch.Export.table.toAsset(
        collection=samples,
        description="S2_RF_Samples_Fixed5Pts_HY2025",
        assetId=S2_FIXED_SAMPLE_ASSET,
    )
    task.start()
    LOGGER.info("Started fixed-point sample-table export: %s", task.id)
    wait_for_task(task, "S2 fixed-point sample table")
    return ee.FeatureCollection(S2_FIXED_SAMPLE_ASSET)


def clean_sample_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = PREDICTOR_BANDS + [CLASS_FIELD, FEATURE_ID_FIELD, FOLD_FIELD]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Local RF table is missing required columns: {missing}")

    out = df.copy()
    for col in PREDICTOR_BANDS + [CLASS_FIELD, FEATURE_ID_FIELD, FOLD_FIELD]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if SAMPLE_ID_FIELD in out.columns:
        out[SAMPLE_ID_FIELD] = pd.to_numeric(out[SAMPLE_ID_FIELD], errors="coerce")

    before = len(out)
    out = out.dropna(subset=required).copy()
    out[CLASS_FIELD] = out[CLASS_FIELD].astype(int)
    out[FEATURE_ID_FIELD] = out[FEATURE_ID_FIELD].astype(int)
    out[FOLD_FIELD] = out[FOLD_FIELD].astype(int)
    if SAMPLE_ID_FIELD in out.columns:
        out[SAMPLE_ID_FIELD] = out[SAMPLE_ID_FIELD].astype("Int64")

    folds = sorted(out[FOLD_FIELD].unique().tolist())
    if folds != list(range(K_FOLDS)):
        raise ValueError(f"Expected cv_fold values 0-{K_FOLDS - 1}; found {folds}")

    LOGGER.info("Sample QA: %d input rows; %d valid rows.", before, len(out))
    LOGGER.info("Independent polygons represented: %d", out[FEATURE_ID_FIELD].nunique())
    return out


def load_or_create_local_samples(sample_asset: ee.FeatureCollection) -> pd.DataFrame:
    RF_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    if LOCAL_SAMPLE_CSV.exists() and not REBUILD_LOCAL_SAMPLE_TABLE:
        LOGGER.info("Loading local fixed-point RF table: %s", LOCAL_SAMPLE_CSV)
        return clean_sample_dataframe(pd.read_csv(LOCAL_SAMPLE_CSV))

    keep = PREDICTOR_BANDS + SAMPLE_PROPERTIES
    LOGGER.info("Downloading the stored 17,175-row sample table once to pandas.")
    df = geemap.ee_to_df(sample_asset.select(keep, retainGeometry=False))
    df = clean_sample_dataframe(df)
    df.to_csv(LOCAL_SAMPLE_CSV, index=False)
    try:
        df.to_excel(LOCAL_SAMPLE_XLSX, index=False)
    except Exception as exc:
        LOGGER.warning("Could not write sample-table Excel file: %s", exc)
    LOGGER.info("Saved local RF sample table: %s", LOCAL_SAMPLE_CSV)
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

        fold_precision, fold_recall, fold_f1, _ = precision_recall_fscore_support(
            valid_df[CLASS_FIELD], pred, labels=CLASS_IDS, zero_division=0
        )
        del fold_precision, fold_recall
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
        LOGGER.info("Running 5-fold CV: %s", model_name)
        result = evaluate_rf_config(samples_df, model_name, config)
        results[model_name] = result
        summaries.append(result["summary"])
        folds.append(result["folds"])
        per_class.append(result["per_class"])
        polygon_summaries.append(result["polygon_summary"])
        polygon_per_class.append(result["polygon_per_class"])
        importances.append(result["importance"])
        result["confusion"].to_csv(
            ACCURACY_DIR / f"S2_CV_ConfusionMatrix_{model_name}.csv"
        )
        result["oof"].to_csv(
            ACCURACY_DIR / f"S2_CV_OOF_{model_name}.csv", index=False
        )

    summary_df = pd.DataFrame(summaries).sort_values("Macro_F1", ascending=False)
    folds_df = pd.concat(folds, ignore_index=True)
    per_class_df = pd.concat(per_class, ignore_index=True)
    polygon_summary_df = pd.DataFrame(polygon_summaries).sort_values(
        "Polygon_Macro_F1", ascending=False
    )
    polygon_per_class_df = pd.concat(polygon_per_class, ignore_index=True)
    importance_df = pd.concat(importances, ignore_index=True)

    summary_df.to_csv(ACCURACY_DIR / "S2_RF_Tuning_5FoldCV_Summary.csv", index=False)
    folds_df.to_csv(ACCURACY_DIR / "S2_RF_Tuning_5FoldCV_Folds.csv", index=False)
    per_class_df.to_csv(ACCURACY_DIR / "S2_RF_Tuning_5FoldCV_PerClass.csv", index=False)
    polygon_summary_df.to_csv(
        ACCURACY_DIR / "S2_RF_Tuning_PolygonMajority_Summary.csv", index=False
    )
    polygon_per_class_df.to_csv(
        ACCURACY_DIR / "S2_RF_Tuning_PolygonMajority_PerClass.csv", index=False
    )
    importance_df.to_csv(ACCURACY_DIR / "S2_RF_Tuning_VariableImportance.csv", index=False)

    LOGGER.info("Five-fold tuning table:\n%s", summary_df.round(3).to_string(index=False))
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
    if asset_exists(S2_CLASSIFIER_ASSET) and not RETRAIN_CLASSIFIER:
        LOGGER.info("Reusing classifier asset: %s", S2_CLASSIFIER_ASSET)
        return ee.Classifier.load(S2_CLASSIFIER_ASSET)

    if asset_exists(S2_CLASSIFIER_ASSET):
        delete_asset(S2_CLASSIFIER_ASSET)

    classifier = build_gee_classifier(samples)
    task = ee.batch.Export.classifier.toAsset(
        classifier=classifier,
        description="S2_RF_Classifier_Fixed5Pts_HY2025_T500",
        assetId=S2_CLASSIFIER_ASSET,
    )
    task.start()
    LOGGER.info("Started 500-tree classifier export: %s", task.id)
    if WAIT_FOR_CLASSIFIER:
        wait_for_task(task, "S2 500-tree classifier")
        return ee.Classifier.load(S2_CLASSIFIER_ASSET)

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
    if asset_exists(S2_TRAINING_PREDICTIONS_ASSET) and not REBUILD_GEE_TRAINING_PREDICTIONS:
        LOGGER.info(
            "Reusing GEE training-prediction asset: %s",
            S2_TRAINING_PREDICTIONS_ASSET,
        )
        return ee.FeatureCollection(S2_TRAINING_PREDICTIONS_ASSET)

    if asset_exists(S2_TRAINING_PREDICTIONS_ASSET):
        delete_asset(S2_TRAINING_PREDICTIONS_ASSET)

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
        description="S2_RF_TrainingPredictions_Fixed5Pts_HY2025_T500",
        assetId=S2_TRAINING_PREDICTIONS_ASSET,
    )
    task.start()
    LOGGER.info("Started batch GEE training-prediction export: %s", task.id)
    wait_for_task(task, "S2 GEE training predictions")
    return ee.FeatureCollection(S2_TRAINING_PREDICTIONS_ASSET)


def get_compact_gee_explain(classifier: ee.Classifier) -> Tuple[float, pd.DataFrame]:
    """Request only small OOB/importance fields; failure is non-fatal."""
    oob_error = np.nan
    importance_df = pd.DataFrame(columns=["Predictor", "Importance", "Importance_percent"])
    try:
        explain = ee.Dictionary(classifier.explain())
        oob_value = explain.get("outOfBagErrorEstimate").getInfo()
        if oob_value is not None:
            oob_error = float(oob_value)
    except Exception as exc:
        LOGGER.warning("Could not retrieve GEE OOB error: %s", exc)

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
        LOGGER.warning("Could not retrieve GEE variable importance: %s", exc)

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
        pred_df.to_csv(ACCURACY_DIR / "S2_GEE_TrainingPredictions.csv", index=False)

        cm_df, class_df, train_metrics = confusion_metrics_from_df(
            pred_df[CLASS_FIELD], pred_df["prediction"]
        )
        cm_df.to_csv(ACCURACY_DIR / "S2_GEE_Training_ConfusionMatrix.csv")
        class_df.to_csv(ACCURACY_DIR / "S2_GEE_Training_PerClass.csv", index=False)
    except Exception as exc:
        LOGGER.warning(
            "GEE resubstitution diagnostics failed; annual maps will still continue: %s",
            exc,
        )

    oob_error, importance_df = get_compact_gee_explain(classifier)
    if not importance_df.empty:
        importance_df.to_csv(ACCURACY_DIR / "S2_GEE_VariableImportance.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "Model": "GEE_SmileRF_Features6_Leaf2",
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
    summary.to_csv(ACCURACY_DIR / "S2_GEE_Accuracy_Summary.csv", index=False)
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
                "Role": "Primary independent validation for HY2025 map model",
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
                "Role": "Complementary validation at source-polygon level",
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
                "Role": "Internal sample-level diagnostic; not independent polygon CV",
            },
        ]
    )
    overview.to_csv(ACCURACY_DIR / "S2_Accuracy_Overview.csv", index=False)

    if EXPORT_ACCURACY_EXCEL:
        workbook = ACCURACY_DIR / "S2_Accuracy_Tables_HY2025.xlsx"
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
            LOGGER.info("Saved consolidated accuracy workbook: %s", workbook)
        except Exception as exc:
            LOGGER.warning("Could not write accuracy Excel workbook: %s", exc)

    return overview


# =============================================================================
# 8. ANNUAL CLASSIFICATION AND EXPORTS
# =============================================================================


def annual_map_asset_id(year: int) -> str:
    return f"{ASSET_ROOT}/Eastern_Morocco_Habitat_S2_RF_Fixed5Pts_HY{year}_T500"


def annual_valid_obs_asset_id(year: int) -> str:
    return f"{ASSET_ROOT}/Eastern_Morocco_S2_ValidObs_HY{year}"


def classify_year(
    year: int,
    classifier: ee.Classifier,
    roi: ee.Geometry,
    terrain: ee.Image,
) -> Tuple[ee.Image, ee.Image]:
    stack, valid_obs = build_s2_predictor_stack(year, roi, terrain)
    start_date, end_date = habitat_year_dates(year)
    classified = (
        stack.select(PREDICTOR_BANDS)
        .classify(classifier)
        .rename("class_id")
        .toUint8()
        .clip(roi)
        .set(
            {
                "sensor": "Sentinel-2",
                "habitat_year": year,
                "start_date": start_date,
                "end_date_exclusive": end_date,
                "training_habitat_year": TRAINING_HABITAT_YEAR,
                "classifier_asset": S2_CLASSIFIER_ASSET,
                "trees": RF_TREES,
                "variables_per_split": GEE_VARIABLES_PER_SPLIT,
                "min_leaf_population": GEE_MIN_LEAF_POPULATION,
                "bag_fraction": GEE_BAG_FRACTION,
            }
        )
    )
    return classified, valid_obs.clip(roi)


def ensure_map_asset(year: int, classified: ee.Image, roi: ee.Geometry) -> Tuple[ee.Image, str]:
    asset_id = annual_map_asset_id(year)
    if not EXPORT_MAPS_TO_ASSET:
        return classified, "not_requested"

    if asset_exists(asset_id) and not OVERWRITE_MAP_ASSETS:
        LOGGER.info("Reusing annual map asset: %s", asset_id)
        return ee.Image(asset_id), "existing"

    if asset_exists(asset_id):
        delete_asset(asset_id)

    task = ee.batch.Export.image.toAsset(
        image=classified,
        description=f"Eastern_Morocco_Habitat_S2_RF_HY{year}_T500",
        assetId=asset_id,
        region=roi,
        crs=EXPORT_CRS,
        crsTransform=EXPORT_TRANSFORM,
        maxPixels=MAX_PIXELS,
        pyramidingPolicy={".default": "mode"},
    )
    task.start()
    LOGGER.info("Started annual map asset HY%d: %s", year, task.id)
    if WAIT_FOR_MAP_ASSET:
        wait_for_task(task, f"S2 map asset HY{year}")
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
        description=f"Eastern_Morocco_S2_ValidObs_HY{year}",
        assetId=asset_id,
        region=roi,
        crs=EXPORT_CRS,
        crsTransform=EXPORT_TRANSFORM,
        maxPixels=MAX_PIXELS,
        pyramidingPolicy={".default": "mean"},
    )
    task.start()
    wait_for_task(task, f"S2 valid observations HY{year}")
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
    name = f"Eastern_Morocco_Habitat_S2_RF_HY{year}_T500"
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
    LOGGER.info("Started Google Drive export HY%d: %s", year, task.id)
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
    name = f"Eastern_Morocco_S2_ValidObs_HY{year}"
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
        LOGGER.info("=== Sentinel-2 HY%d ===", year)
        start_date, end_date = habitat_year_dates(year)
        scene_count = int(get_s2_collection(year, roi).size().getInfo())
        if scene_count == 0:
            LOGGER.warning("HY%d has no filtered Sentinel-2 scenes; skipping.", year)
            rows.append(
                {
                    "habitat_year": year,
                    "start_date": start_date,
                    "end_date_exclusive": end_date,
                    "scene_count": scene_count,
                    "status": "skipped_no_scenes",
                }
            )
            continue

        classified, valid_obs = classify_year(year, classifier, roi, terrain)
        map_image, asset_status = ensure_map_asset(year, classified, roi)
        obs_status = ensure_valid_obs_asset(year, valid_obs, roi)
        drive_task = start_drive_export(year, map_image, roi, drive_tasks)
        obs_drive_task = start_valid_obs_drive_export(year, valid_obs, roi, drive_tasks)

        row = {
            "sensor": "Sentinel-2",
            "habitat_year": year,
            "start_date": start_date,
            "end_date_exclusive": end_date,
            "scene_count": scene_count,
            "map_asset": annual_map_asset_id(year) if EXPORT_MAPS_TO_ASSET else "",
            "map_asset_status": asset_status,
            "drive_folder": DRIVE_FOLDER if EXPORT_MAPS_TO_DRIVE else "",
            "drive_task_id": drive_task.id if drive_task is not None else "",
            "valid_obs_asset_status": obs_status,
            "valid_obs_drive_task_id": obs_drive_task.id if obs_drive_task else "",
            "training_habitat_year": TRAINING_HABITAT_YEAR,
            "validation_scope": "HY2025 polygon-grouped 5-fold CV; annual temporal transfer",
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(
            MANIFEST_DIR / "S2_Annual_Export_Manifest.csv", index=False
        )

    if WAIT_FOR_DRIVE_AT_END:
        wait_for_drive_tasks(drive_tasks)

    # Refresh task states without forcing the script to wait for Drive by default.
    task_states = {
        name: str(task.status().get("state", "UNKNOWN"))
        for name, task in drive_tasks.items()
    }
    manifest = pd.DataFrame(rows)
    if not manifest.empty and "habitat_year" in manifest.columns:
        manifest["drive_task_state_at_script_end"] = manifest["habitat_year"].map(
            lambda y: task_states.get(f"Eastern_Morocco_Habitat_S2_RF_HY{int(y)}_T500", "")
        )
    manifest.to_csv(MANIFEST_DIR / "S2_Annual_Export_Manifest.csv", index=False)
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
        "hy2025_spectral_asset": S2_HY2025_SPECTRAL_ASSET,
        "hy2025_predictor_asset": S2_HY2025_PREDICTOR_ASSET,
        "fixed_sample_asset": S2_FIXED_SAMPLE_ASSET,
        "classifier_asset": S2_CLASSIFIER_ASSET,
        "training_prediction_asset": S2_TRAINING_PREDICTIONS_ASSET,
        "local_sample_csv": str(LOCAL_SAMPLE_CSV),
        "habitat_years": HABITAT_YEARS,
        "predictor_bands": PREDICTOR_BANDS,
        "cloud_score_threshold": CLEAR_THRESHOLD,
        "rf_trees": RF_TREES,
        "variables_per_split": GEE_VARIABLES_PER_SPLIT,
        "min_leaf_population": GEE_MIN_LEAF_POPULATION,
        "bag_fraction": GEE_BAG_FRACTION,
        "random_seed": RANDOM_SEED,
        "export_crs": EXPORT_CRS,
        "export_transform": EXPORT_TRANSFORM,
        "class_map": CLASS_MAP,
    }
    with (MANIFEST_DIR / "S2_Run_Configuration.json").open("w", encoding="utf-8") as fp:
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

    # 1) Materialize the fixed-point HY2025 predictor table once, then use the
    # local CSV for all sklearn computations.
    sample_asset = ensure_fixed_sample_asset()
    samples_df = load_or_create_local_samples(sample_asset)

    # 2) Reproduce the full six-configuration tuning table with the same folds.
    cv_summary, cv_results = run_and_save_cv(samples_df)

    # 3) Train or reuse the final 500-tree GEE classifier.
    classifier = ensure_classifier_asset(sample_asset)

    # 4) Obtain safe GEE training/OOB diagnostics without a large interactive
    # confusion-matrix request.
    gee_summary, gee_cm, gee_per_class, gee_importance = save_gee_diagnostics(
        classifier, sample_asset
    )

    # 5) Save a manuscript-friendly overview and workbook.
    overview = save_accuracy_overview(
        cv_summary,
        cv_results,
        gee_summary,
        gee_cm,
        gee_per_class,
        gee_importance,
    )
    LOGGER.info("Accuracy overview:\n%s", overview.round(3).to_string(index=False))

    # 6) Export the annual HY2018-HY2025 habitat maps.
    manifest = run_annual_exports(classifier, roi, terrain)
    save_run_configuration()

    LOGGER.info("Sentinel-2 production workflow finished.")
    LOGGER.info("Accuracy tables: %s", ACCURACY_DIR)
    LOGGER.info("Export manifest: %s", MANIFEST_DIR / "S2_Annual_Export_Manifest.csv")
    LOGGER.info("Google Drive folder: %s", DRIVE_FOLDER)
    if not WAIT_FOR_DRIVE_AT_END and EXPORT_MAPS_TO_DRIVE:
        LOGGER.info("Drive tasks continue on Earth Engine after the Python script exits.")


if __name__ == "__main__":
    main()
