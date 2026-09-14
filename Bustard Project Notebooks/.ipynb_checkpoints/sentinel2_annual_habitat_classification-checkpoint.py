#!/usr/bin/env python
"""
Annual Sentinel-2 habitat classification for Eastern Morocco.

Workflow
--------
1. Initialize Google Earth Engine.
2. Build or load the HY2025 predictor-sample table.
3. Run polygon-grouped five-fold Random Forest cross-validation locally.
4. Train a 500-tree Earth Engine Smile Random Forest classifier.
5. Save accuracy tables and variable importance locally.
6. Export annual habitat maps to Earth Engine Assets and/or Google Drive.

Habitat year YYYY is defined as 1 September YYYY-1 through 31 August YYYY.
The default annual period is HY2018-HY2025 because HY2018 is the first
complete September-August year in the Sentinel-2 L2A surface-reflectance
archive.

Important interpretation
------------------------
The accuracy assessment uses HY2025 reference data. Accuracy values are model-
level estimates for the reference year, not independently measured accuracy
for every historical annual map. Application to other years is temporal model
transfer and should be checked with temporally matched reference imagery.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
# USER CONFIGURATION
# =============================================================================

EE_PROJECT = "rse-global-wetlands"
ASSET_ROOT = "projects/rse-global-wetlands/assets"

ROI_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_Working_Area"
TRAINING_POINTS_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_RF_Points_HY2025"
TERRAIN_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_Terrain_GLO30"

# Existing sample asset from the completed Sentinel-2 workflow.
RF_SAMPLE_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_RF_Samples_HY2025_5px"
CLASSIFIER_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_S2_RF_Classifier_HY2025_T500"

TRAINING_HABITAT_YEAR = 2025
HABITAT_YEARS = list(range(2018, 2026))

# Local output folder for accuracy tables and run manifests.
LOCAL_OUTPUT_DIR = Path(
    r"C:\Users\PangY\OneDrive - Smithsonian Institution\Bustard"
    r"\01_Data\Habitat mapping\Annual_Classification\Sentinel2"
)

DRIVE_FOLDER = "Eastern_Morocco_Habitat_Sentinel2_Annual"

# Export choices.
EXPORT_MAPS_TO_ASSET = True
EXPORT_MAPS_TO_DRIVE = True
EXPORT_PREDICTOR_STACKS_TO_ASSET = False
EXPORT_VALID_OBSERVATION_COUNT_TO_ASSET = False

# Existing assets are reused by default. Set True only when intentionally
# replacing products with the same IDs.
OVERWRITE_ASSETS = False

# If True, the script waits for classifier and image-asset exports. Google
# Drive exports may take hours; they are launched but not waited for by default.
WAIT_FOR_CLASSIFIER_EXPORT = True
WAIT_FOR_MAP_ASSET_EXPORTS = True
WAIT_FOR_DRIVE_EXPORTS = False
POLL_SECONDS = 60

# Sentinel-2 processing.
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
QA_BAND = "cs_cdf"
CLEAR_THRESHOLD = 0.60
MAX_SCENE_CLOUD_PERCENT = 80

# Model predictors.
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

CLASS_PALETTE = [
    "B8D96B",  # 0 Grass steppe
    "6BAF5E",  # 1 Shrub steppe
    "2E7D32",  # 2 Wooded steppe
    "8C8C8C",  # 3 Bare rocky
    "D8C07A",  # 4 Spreading area
    "C9B458",  # 5 Salty steppe
    "E8C547",  # 6 Dune
    "5AA7D9",  # 7 Wadi and gullies
    "E89A61",  # 8 Cultivated field/fallow
    "2166AC",  # 9 Water body
    "D73027",  # 10 Built-up
]

# Tuned/selected RF settings.
RANDOM_SEED = 42
K_FOLDS = 5
LOCAL_RF_TREES = 500
GEE_RF_TREES = 500
MAX_FEATURES = 6
MIN_SAMPLES_LEAF = 2
GEE_BAG_FRACTION = 0.632

# Export grid. A fixed transform is used so every annual product is aligned.
EXPORT_CRS = "EPSG:32630"
EXPORT_SCALE = 10
EXPORT_TRANSFORM = [10, 0, 0, 0, -10, 10_000_000]
MAX_PIXELS = 1e13
DRIVE_NODATA = 255


# =============================================================================
# LOGGING AND EARTH ENGINE HELPERS
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

TERMINAL_TASK_STATES = {"COMPLETED", "FAILED", "CANCELLED", "CANCEL_REQUESTED"}


def initialize_ee() -> None:
    """Authenticate if necessary and initialize Earth Engine."""
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        LOGGER.info("Earth Engine initialization failed; starting authentication.")
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)
    LOGGER.info("Earth Engine initialized with project %s", EE_PROJECT)


def asset_exists(asset_id: str) -> bool:
    """Return True when an Earth Engine asset is accessible."""
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def remove_asset_if_requested(asset_id: str) -> None:
    """Delete an existing asset only when OVERWRITE_ASSETS is enabled."""
    if asset_exists(asset_id):
        if not OVERWRITE_ASSETS:
            raise FileExistsError(
                f"Asset already exists and OVERWRITE_ASSETS=False: {asset_id}"
            )
        LOGGER.warning("Deleting existing asset: %s", asset_id)
        ee.data.deleteAsset(asset_id)


def task_state(task: ee.batch.Task) -> str:
    return str(task.status().get("state", "UNKNOWN"))


def wait_for_tasks(
    tasks: Dict[str, ee.batch.Task],
    poll_seconds: int = POLL_SECONDS,
    fail_on_error: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Poll a dictionary of EE batch tasks until all reach terminal states."""
    if not tasks:
        return {}

    LOGGER.info("Waiting for %d Earth Engine task(s).", len(tasks))
    last_states: Dict[str, str] = {}

    while True:
        statuses = {name: task.status() for name, task in tasks.items()}
        states = {name: str(status.get("state", "UNKNOWN")) for name, status in statuses.items()}

        changed = states != last_states
        if changed:
            for name, state in states.items():
                LOGGER.info("Task %-45s %s", name, state)
            last_states = states

        if all(state in TERMINAL_TASK_STATES for state in states.values()):
            failures = {
                name: status
                for name, status in statuses.items()
                if str(status.get("state")) != "COMPLETED"
            }
            if failures and fail_on_error:
                details = "\n".join(
                    f"{name}: {status.get('state')} - {status.get('error_message', '')}"
                    for name, status in failures.items()
                )
                raise RuntimeError(f"One or more Earth Engine tasks failed:\n{details}")
            return statuses

        time.sleep(poll_seconds)


def habitat_year_dates(habitat_year: int) -> Tuple[str, str]:
    """Return inclusive start and exclusive end dates for a habitat year."""
    return f"{habitat_year - 1}-09-01", f"{habitat_year}-09-01"


# =============================================================================
# PREDICTOR GENERATION
# =============================================================================


def prepare_s2_image(image: ee.Image) -> ee.Image:
    """Mask clouds/shadows and scale Sentinel-2 surface reflectance."""
    image = ee.Image(image)
    clear_mask = image.select(QA_BAND).gte(CLEAR_THRESHOLD)
    scl = image.select("SCL")

    # Remove no data, defective pixels, cloud shadow, medium/high cloud,
    # cirrus, and snow/ice.
    scl_mask = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )

    reflectance = (
        image.select(S2_BANDS)
        .multiply(0.0001)
        .updateMask(clear_mask)
        .updateMask(scl_mask)
    )

    return reflectance.copyProperties(
        image,
        ["system:time_start", "system:index", "MGRS_TILE"],
    )


def add_spectral_indices(composite: ee.Image) -> ee.Image:
    """Add EVI, MSAVI, NDWI, and BSI to a scaled reflectance composite."""
    blue = composite.select("B2")
    green = composite.select("B3")
    red = composite.select("B4")
    nir = composite.select("B8")
    swir1 = composite.select("B11")

    evi = composite.expression(
        "2.5 * ((NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0))",
        {"NIR": nir, "RED": red, "BLUE": blue},
    ).rename("EVI")

    msavi_term = nir.multiply(2).add(1)
    msavi_discriminant = (
        msavi_term.pow(2).subtract(nir.subtract(red).multiply(8)).max(0)
    )
    msavi = (
        msavi_term.subtract(msavi_discriminant.sqrt()).divide(2).rename("MSAVI")
    )

    ndwi = composite.normalizedDifference(["B3", "B8"]).rename("NDWI")

    bsi = composite.expression(
        "((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))",
        {"SWIR": swir1, "RED": red, "NIR": nir, "BLUE": blue},
    ).rename("BSI")

    return composite.addBands([evi, msavi, ndwi, bsi])


def build_terrain(roi: ee.Geometry) -> ee.Image:
    """Load the saved terrain asset, or build it from Copernicus GLO-30."""
    if asset_exists(TERRAIN_ASSET):
        LOGGER.info("Using saved terrain asset: %s", TERRAIN_ASSET)
        return ee.Image(TERRAIN_ASSET).select(TERRAIN_BANDS).resample("bilinear")

    LOGGER.warning("Terrain asset not found; building terrain from GLO-30.")
    glo30 = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").filterBounds(roi)
    native_projection = ee.Image(glo30.first()).select("DEM").projection()
    dem = (
        glo30.select("DEM")
        .mosaic()
        .setDefaultProjection(native_projection)
        .rename("Elevation")
    )
    slope = ee.Terrain.slope(dem).rename("Slope")
    aspect_rad = ee.Terrain.aspect(dem).multiply(math.pi / 180.0)
    northness = aspect_rad.cos().rename("Northness")
    eastness = aspect_rad.sin().rename("Eastness")
    return ee.Image.cat([dem, slope, northness, eastness]).resample("bilinear")


def build_s2_predictor_stack(
    habitat_year: int,
    roi: ee.Geometry,
    terrain: ee.Image,
) -> Tuple[ee.Image, ee.Image, ee.ImageCollection]:
    """Build the 14-band Sentinel-2 predictor stack and valid-observation count."""
    start_date, end_date = habitat_year_dates(habitat_year)

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
    clean = s2.linkCollection(cloud_score, [QA_BAND]).map(prepare_s2_image)

    composite = clean.median()
    spectral = add_spectral_indices(composite)
    valid_obs = clean.select("B2").count().rename("valid_obs").toUint16()

    stack = (
        spectral.select(S2_BANDS + INDEX_BANDS)
        .addBands(terrain)
        .select(PREDICTOR_BANDS)
        .float()
        .set(
            {
                "sensor": "Sentinel-2",
                "habitat_year": habitat_year,
                "start_date": start_date,
                "end_date_exclusive": end_date,
                "training_habitat_year": TRAINING_HABITAT_YEAR,
                "predictor_count": len(PREDICTOR_BANDS),
            }
        )
    )
    return stack, valid_obs, clean


# =============================================================================
# SAMPLE TABLE AND LOCAL CROSS-VALIDATION
# =============================================================================


def ensure_sample_asset(roi: ee.Geometry, terrain: ee.Image) -> ee.FeatureCollection:
    """Return the permanent sample asset, creating it when it is absent."""
    if asset_exists(RF_SAMPLE_ASSET):
        LOGGER.info("Using existing RF sample asset: %s", RF_SAMPLE_ASSET)
        return ee.FeatureCollection(RF_SAMPLE_ASSET)

    LOGGER.info("RF sample asset is absent; creating it from training points.")
    points = ee.FeatureCollection(TRAINING_POINTS_ASSET)
    predictor_stack, _, _ = build_s2_predictor_stack(
        TRAINING_HABITAT_YEAR, roi, terrain
    )
    fixed_projection = ee.Projection(EXPORT_CRS).atScale(EXPORT_SCALE)

    samples = predictor_stack.sampleRegions(
        collection=points,
        properties=SAMPLE_PROPERTIES,
        scale=EXPORT_SCALE,
        projection=fixed_projection,
        geometries=False,
        tileScale=4,
    )
    samples = samples.filter(
        ee.Filter.notNull(PREDICTOR_BANDS + [CLASS_FIELD, FOLD_FIELD, FEATURE_ID_FIELD])
    )

    remove_asset_if_requested(RF_SAMPLE_ASSET)
    task = ee.batch.Export.table.toAsset(
        collection=samples,
        description="Eastern_Morocco_S2_RF_Samples_HY2025",
        assetId=RF_SAMPLE_ASSET,
    )
    task.start()
    wait_for_tasks({"S2 sample table": task})
    return ee.FeatureCollection(RF_SAMPLE_ASSET)


def download_samples(rf_samples: ee.FeatureCollection) -> pd.DataFrame:
    """Download a slim sample table and enforce numeric column types."""
    keep_fields = PREDICTOR_BANDS + SAMPLE_PROPERTIES
    slim = rf_samples.select(keep_fields, retainGeometry=False)
    LOGGER.info("Downloading RF sample table to pandas.")
    df = geemap.ee_to_df(slim)

    required = PREDICTOR_BANDS + [CLASS_FIELD, FOLD_FIELD, FEATURE_ID_FIELD]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"RF sample table is missing required fields: {missing}")

    for column in PREDICTOR_BANDS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in [CLASS_FIELD, FOLD_FIELD, FEATURE_ID_FIELD]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if SAMPLE_ID_FIELD in df.columns:
        df[SAMPLE_ID_FIELD] = pd.to_numeric(df[SAMPLE_ID_FIELD], errors="coerce")

    before = len(df)
    df = df.dropna(subset=required).copy()
    for column in [CLASS_FIELD, FOLD_FIELD, FEATURE_ID_FIELD]:
        df[column] = df[column].astype(int)

    LOGGER.info("Downloaded %d samples; %d remain after QA.", before, len(df))
    return df


def majority_vote(values: pd.Series) -> int:
    """Return a deterministic majority class; lowest class wins a tie."""
    counts = values.value_counts()
    return int(sorted(counts[counts == counts.max()].index.tolist())[0])


def evaluate_local_rf(
    samples_df: pd.DataFrame,
    model_name: str,
    class_weight: Optional[str],
) -> Dict[str, Any]:
    """Run fixed polygon-grouped five-fold CV for one RF configuration."""
    fold_rows: List[Dict[str, Any]] = []
    oof_frames: List[pd.DataFrame] = []
    importance_series: List[pd.Series] = []

    for fold in range(K_FOLDS):
        train_df = samples_df[samples_df[FOLD_FIELD] != fold]
        valid_df = samples_df[samples_df[FOLD_FIELD] == fold]
        if valid_df.empty:
            raise ValueError(f"No validation samples found for fold {fold}.")

        model = RandomForestClassifier(
            n_estimators=LOCAL_RF_TREES,
            max_features=MAX_FEATURES,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            class_weight=class_weight,
            bootstrap=True,
            random_state=RANDOM_SEED + fold,
            n_jobs=-1,
        )
        model.fit(train_df[PREDICTOR_BANDS], train_df[CLASS_FIELD])
        prediction = model.predict(valid_df[PREDICTOR_BANDS]).astype(int)

        fold_rows.append(
            {
                "Model": model_name,
                "Fold": fold + 1,
                "N_train": len(train_df),
                "N_validation": len(valid_df),
                "Overall_Accuracy": accuracy_score(valid_df[CLASS_FIELD], prediction),
                "Kappa": cohen_kappa_score(valid_df[CLASS_FIELD], prediction),
                "Balanced_Accuracy": balanced_accuracy_score(
                    valid_df[CLASS_FIELD], prediction
                ),
            }
        )

        id_fields = [FEATURE_ID_FIELD, CLASS_FIELD, FOLD_FIELD]
        if SAMPLE_ID_FIELD in valid_df.columns:
            id_fields.insert(0, SAMPLE_ID_FIELD)
        temp = valid_df[id_fields].copy()
        temp["prediction"] = prediction
        oof_frames.append(temp)
        importance_series.append(
            pd.Series(
                model.feature_importances_,
                index=PREDICTOR_BANDS,
                name=f"Fold_{fold + 1}",
            )
        )

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.concat(oof_frames, ignore_index=True)
    y_true = oof_df[CLASS_FIELD].astype(int)
    y_pred = oof_df["prediction"].astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        zero_division=0,
    )
    class_df = pd.DataFrame(
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
    polygon_precision, polygon_recall, polygon_f1, polygon_support = (
        precision_recall_fscore_support(
            polygon_oof[CLASS_FIELD],
            polygon_oof["prediction"],
            labels=CLASS_IDS,
            zero_division=0,
        )
    )
    polygon_class_df = pd.DataFrame(
        {
            "Model": model_name,
            "class_id": CLASS_IDS,
            "Habitat": [CLASS_MAP[i] for i in CLASS_IDS],
            "N_validation_polygons": polygon_support,
            "Precision_UserAcc": polygon_precision,
            "Recall_ProducerAcc": polygon_recall,
            "F1": polygon_f1,
        }
    )

    summary = {
        "Model": model_name,
        "Trees": LOCAL_RF_TREES,
        "MaxFeatures": MAX_FEATURES,
        "MinSamplesLeaf": MIN_SAMPLES_LEAF,
        "ClassWeight": str(class_weight),
        "Pooled_OA": accuracy_score(y_true, y_pred),
        "Pooled_Kappa": cohen_kappa_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro_F1": float(np.mean(f1)),
        "Fold_OA_Mean": fold_df["Overall_Accuracy"].mean(),
        "Fold_OA_SD": fold_df["Overall_Accuracy"].std(),
        "Fold_Kappa_Mean": fold_df["Kappa"].mean(),
        "Fold_Kappa_SD": fold_df["Kappa"].std(),
        "Polygon_OA": accuracy_score(
            polygon_oof[CLASS_FIELD], polygon_oof["prediction"]
        ),
        "Polygon_Kappa": cohen_kappa_score(
            polygon_oof[CLASS_FIELD], polygon_oof["prediction"]
        ),
        "Polygon_Balanced_Accuracy": balanced_accuracy_score(
            polygon_oof[CLASS_FIELD], polygon_oof["prediction"]
        ),
        "Polygon_Macro_F1": float(np.mean(polygon_f1)),
    }

    importance_matrix = pd.concat(importance_series, axis=1)
    importance_df = pd.DataFrame(
        {
            "Model": model_name,
            "Predictor": PREDICTOR_BANDS,
            "Importance_mean": importance_matrix.mean(axis=1).reindex(PREDICTOR_BANDS).values,
            "Importance_SD": importance_matrix.std(axis=1).reindex(PREDICTOR_BANDS).values,
        }
    ).sort_values("Importance_mean", ascending=False)

    return {
        "summary": summary,
        "folds": fold_df,
        "classes": class_df,
        "confusion": cm_df,
        "oof": oof_df,
        "polygon_classes": polygon_class_df,
        "importance": importance_df,
    }


def save_local_cv_results(samples_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate map-matched and balanced sensitivity models and save tables."""
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples_df.to_csv(LOCAL_OUTPUT_DIR / "S2_RF_Samples_HY2025.csv", index=False)

    configs = {
        "Features6_Leaf2_MapMatched": None,
        "Balanced_Features6_Leaf2_Sensitivity": "balanced_subsample",
    }

    results: Dict[str, Any] = {}
    summaries: List[Dict[str, Any]] = []
    all_folds: List[pd.DataFrame] = []
    all_classes: List[pd.DataFrame] = []
    all_polygon_classes: List[pd.DataFrame] = []
    all_importance: List[pd.DataFrame] = []

    for name, weight in configs.items():
        LOGGER.info("Running local five-fold CV: %s", name)
        result = evaluate_local_rf(samples_df, name, weight)
        results[name] = result
        summaries.append(result["summary"])
        all_folds.append(result["folds"])
        all_classes.append(result["classes"])
        all_polygon_classes.append(result["polygon_classes"])
        all_importance.append(result["importance"])
        result["confusion"].to_csv(
            LOCAL_OUTPUT_DIR / f"S2_CV_ConfusionMatrix_{name}.csv"
        )
        result["oof"].to_csv(
            LOCAL_OUTPUT_DIR / f"S2_CV_OutOfFoldPredictions_{name}.csv",
            index=False,
        )

    summary_df = pd.DataFrame(summaries).sort_values("Macro_F1", ascending=False)
    summary_df.to_csv(LOCAL_OUTPUT_DIR / "S2_CV_Model_Summary.csv", index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(
        LOCAL_OUTPUT_DIR / "S2_CV_Fold_Metrics.csv", index=False
    )
    pd.concat(all_classes, ignore_index=True).to_csv(
        LOCAL_OUTPUT_DIR / "S2_CV_PerClass_Metrics.csv", index=False
    )
    pd.concat(all_polygon_classes, ignore_index=True).to_csv(
        LOCAL_OUTPUT_DIR / "S2_CV_Polygon_PerClass_Metrics.csv", index=False
    )
    pd.concat(all_importance, ignore_index=True).to_csv(
        LOCAL_OUTPUT_DIR / "S2_CV_Variable_Importance.csv", index=False
    )

    LOGGER.info("Saved local CV tables to %s", LOCAL_OUTPUT_DIR)
    return summary_df, results


# =============================================================================
# EARTH ENGINE CLASSIFIER AND DIAGNOSTICS
# =============================================================================


def training_metrics_from_matrix(cm: np.ndarray) -> Tuple[pd.DataFrame, float, float]:
    """Return per-class metrics, balanced accuracy, and macro F1."""
    tp = np.diag(cm)
    actual_total = cm.sum(axis=1)
    predicted_total = cm.sum(axis=0)
    recall = np.divide(
        tp,
        actual_total,
        out=np.zeros(len(tp), dtype=float),
        where=actual_total != 0,
    )
    precision = np.divide(
        tp,
        predicted_total,
        out=np.zeros(len(tp), dtype=float),
        where=predicted_total != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(tp), dtype=float),
        where=(precision + recall) != 0,
    )
    class_df = pd.DataFrame(
        {
            "class_id": CLASS_IDS,
            "Habitat": [CLASS_MAP[i] for i in CLASS_IDS],
            "Precision_UserAcc": precision,
            "Recall_ProducerAcc": recall,
            "F1": f1,
            "Support": actual_total,
        }
    )
    return class_df, float(np.mean(recall)), float(np.mean(f1))


def build_gee_classifier(rf_samples: ee.FeatureCollection) -> ee.Classifier:
    """Train the operational Earth Engine classifier with selected settings."""
    clean_samples = rf_samples.filter(
        ee.Filter.notNull(PREDICTOR_BANDS + [CLASS_FIELD])
    )
    return ee.Classifier.smileRandomForest(
        numberOfTrees=GEE_RF_TREES,
        variablesPerSplit=MAX_FEATURES,
        minLeafPopulation=MIN_SAMPLES_LEAF,
        bagFraction=GEE_BAG_FRACTION,
        seed=RANDOM_SEED,
    ).train(
        features=clean_samples,
        classProperty=CLASS_FIELD,
        inputProperties=PREDICTOR_BANDS,
    )


def save_gee_diagnostics(classifier: ee.Classifier) -> pd.DataFrame:
    """Calculate and save GEE training/OOB diagnostics."""
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cm_ee = classifier.confusionMatrix()
    cm = np.asarray(cm_ee.array().getInfo(), dtype=np.int64)
    train_oa = float(cm_ee.accuracy().getInfo())
    train_kappa = float(cm_ee.kappa().getInfo())

    class_df, balanced_accuracy, macro_f1 = training_metrics_from_matrix(cm)
    class_df.to_csv(
        LOCAL_OUTPUT_DIR / "S2_GEE_Training_PerClass_Metrics.csv", index=False
    )

    cm_df = pd.DataFrame(
        cm,
        index=[CLASS_MAP[i] for i in CLASS_IDS],
        columns=[CLASS_MAP[i] for i in CLASS_IDS],
    )
    cm_df.index.name = "Actual"
    cm_df.columns.name = "Predicted"
    cm_df.to_csv(LOCAL_OUTPUT_DIR / "S2_GEE_Training_ConfusionMatrix.csv")

    oob_error = np.nan
    importance: Dict[str, float] = {}
    try:
        explanation = classifier.explain().getInfo()
        oob_value = explanation.get("outOfBagErrorEstimate")
        if oob_value is not None:
            oob_error = float(oob_value)
        importance = {
            str(k): float(v) for k, v in explanation.get("importance", {}).items()
        }
    except Exception as exc:
        LOGGER.warning("Could not retrieve full GEE classifier explanation: %s", exc)

    if importance:
        importance_df = pd.DataFrame(
            {"Predictor": list(importance.keys()), "Importance": list(importance.values())}
        ).sort_values("Importance", ascending=False)
        importance_df["Importance_percent"] = (
            100.0 * importance_df["Importance"] / importance_df["Importance"].sum()
        )
        importance_df.to_csv(
            LOCAL_OUTPUT_DIR / "S2_GEE_Variable_Importance.csv", index=False
        )

    summary_df = pd.DataFrame(
        [
            {
                "Model": "GEE_SmileRF_Features6_Leaf2",
                "Trees": GEE_RF_TREES,
                "VariablesPerSplit": MAX_FEATURES,
                "MinLeafPopulation": MIN_SAMPLES_LEAF,
                "BagFraction": GEE_BAG_FRACTION,
                "Training_OA": train_oa,
                "Training_Kappa": train_kappa,
                "Training_Balanced_Accuracy": balanced_accuracy,
                "Training_Macro_F1": macro_f1,
                "OOB_Error": oob_error,
                "Approx_OOB_Accuracy": 1.0 - oob_error if np.isfinite(oob_error) else np.nan,
            }
        ]
    )
    summary_df.to_csv(LOCAL_OUTPUT_DIR / "S2_GEE_Accuracy_Summary.csv", index=False)
    LOGGER.info("Saved GEE diagnostic tables to %s", LOCAL_OUTPUT_DIR)
    return summary_df


def ensure_classifier_asset(classifier: ee.Classifier) -> ee.Classifier:
    """Export and load the trained classifier, or reuse the existing asset."""
    if asset_exists(CLASSIFIER_ASSET) and not OVERWRITE_ASSETS:
        LOGGER.info("Using existing classifier asset: %s", CLASSIFIER_ASSET)
        return ee.Classifier.load(CLASSIFIER_ASSET)

    remove_asset_if_requested(CLASSIFIER_ASSET)
    task = ee.batch.Export.classifier.toAsset(
        classifier=classifier,
        description="Eastern_Morocco_S2_RF_Classifier_HY2025_T500",
        assetId=CLASSIFIER_ASSET,
    )
    task.start()
    LOGGER.info("Started classifier export: %s", task.id)

    if WAIT_FOR_CLASSIFIER_EXPORT:
        wait_for_tasks({"S2 classifier": task})
        return ee.Classifier.load(CLASSIFIER_ASSET)

    LOGGER.warning(
        "Classifier export is still asynchronous. The in-memory classifier will be used."
    )
    return classifier


# =============================================================================
# ANNUAL MAP EXPORTS
# =============================================================================


def annual_map_asset_id(habitat_year: int) -> str:
    return f"{ASSET_ROOT}/Eastern_Morocco_Habitat_S2_RF_HY{habitat_year}"


def predictor_asset_id(habitat_year: int) -> str:
    return f"{ASSET_ROOT}/Eastern_Morocco_S2_PredictorStack_HY{habitat_year}"


def observation_asset_id(habitat_year: int) -> str:
    return f"{ASSET_ROOT}/Eastern_Morocco_S2_ValidObs_HY{habitat_year}"


def export_annual_assets(
    classifier: ee.Classifier,
    roi: ee.Geometry,
    terrain: ee.Image,
) -> Tuple[Dict[str, ee.batch.Task], List[Dict[str, Any]]]:
    """Launch annual classification, predictor, and QA asset exports."""
    tasks: Dict[str, ee.batch.Task] = {}
    manifest: List[Dict[str, Any]] = []

    for year in HABITAT_YEARS:
        LOGGER.info("Preparing Sentinel-2 HY%d", year)
        stack, valid_obs, collection = build_s2_predictor_stack(year, roi, terrain)
        scene_count = int(collection.size().getInfo())
        LOGGER.info("HY%d contains %d filtered Sentinel-2 scenes.", year, scene_count)

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
                    "training_habitat_year": TRAINING_HABITAT_YEAR,
                    "classifier_asset": CLASSIFIER_ASSET,
                    "trees": GEE_RF_TREES,
                    "variables_per_split": MAX_FEATURES,
                    "min_leaf_population": MIN_SAMPLES_LEAF,
                    "bag_fraction": GEE_BAG_FRACTION,
                }
            )
        )

        map_asset = annual_map_asset_id(year)
        map_status = "not_requested"
        if EXPORT_MAPS_TO_ASSET:
            if asset_exists(map_asset) and not OVERWRITE_ASSETS:
                LOGGER.info("Map asset already exists; reusing %s", map_asset)
                map_status = "existing"
            else:
                remove_asset_if_requested(map_asset)
                name = f"S2_Habitat_RF_HY{year}"
                task = ee.batch.Export.image.toAsset(
                    image=classified,
                    description=name,
                    assetId=map_asset,
                    region=roi,
                    crs=EXPORT_CRS,
                    crsTransform=EXPORT_TRANSFORM,
                    maxPixels=MAX_PIXELS,
                    pyramidingPolicy={".default": "mode"},
                )
                task.start()
                tasks[name] = task
                map_status = "submitted"

        if EXPORT_PREDICTOR_STACKS_TO_ASSET:
            pred_asset = predictor_asset_id(year)
            if not (asset_exists(pred_asset) and not OVERWRITE_ASSETS):
                remove_asset_if_requested(pred_asset)
                name = f"S2_PredictorStack_HY{year}"
                task = ee.batch.Export.image.toAsset(
                    image=stack.clip(roi),
                    description=name,
                    assetId=pred_asset,
                    region=roi,
                    crs=EXPORT_CRS,
                    crsTransform=EXPORT_TRANSFORM,
                    maxPixels=MAX_PIXELS,
                    pyramidingPolicy={".default": "mean"},
                )
                task.start()
                tasks[name] = task

        if EXPORT_VALID_OBSERVATION_COUNT_TO_ASSET:
            obs_asset = observation_asset_id(year)
            if not (asset_exists(obs_asset) and not OVERWRITE_ASSETS):
                remove_asset_if_requested(obs_asset)
                name = f"S2_ValidObs_HY{year}"
                task = ee.batch.Export.image.toAsset(
                    image=valid_obs.clip(roi),
                    description=name,
                    assetId=obs_asset,
                    region=roi,
                    crs=EXPORT_CRS,
                    crsTransform=EXPORT_TRANSFORM,
                    maxPixels=MAX_PIXELS,
                    pyramidingPolicy={".default": "mean"},
                )
                task.start()
                tasks[name] = task

        start_date, end_date = habitat_year_dates(year)
        manifest.append(
            {
                "sensor": "Sentinel-2",
                "habitat_year": year,
                "start_date": start_date,
                "end_date_exclusive": end_date,
                "scene_count": scene_count,
                "map_asset": map_asset,
                "asset_status_at_submission": map_status,
                "training_habitat_year": TRAINING_HABITAT_YEAR,
                "validation_scope": "HY2025 grouped polygon CV; annual temporal transfer not independently validated",
            }
        )

    return tasks, manifest


def export_maps_to_drive(roi: ee.Geometry) -> Dict[str, ee.batch.Task]:
    """Export completed annual asset maps to Google Drive as COG GeoTIFFs."""
    tasks: Dict[str, ee.batch.Task] = {}
    if not EXPORT_MAPS_TO_DRIVE:
        return tasks

    for year in HABITAT_YEARS:
        asset_id = annual_map_asset_id(year)
        if not asset_exists(asset_id):
            LOGGER.warning("Skipping Drive export; map asset is absent: %s", asset_id)
            continue

        name = f"Eastern_Morocco_Habitat_S2_RF_HY{year}"
        image = ee.Image(asset_id).unmask(value=DRIVE_NODATA, sameFootprint=False).toUint8()
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
            formatOptions={"cloudOptimized": True, "noData": DRIVE_NODATA},
            skipEmptyTiles=True,
        )
        task.start()
        tasks[name] = task
        LOGGER.info("Started Drive export for HY%d: %s", year, task.id)

    return tasks


def save_run_metadata(
    cv_summary: pd.DataFrame,
    gee_summary: pd.DataFrame,
    manifest: List[Dict[str, Any]],
) -> None:
    """Save annual manifest and model settings locally."""
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    map_matched = cv_summary[
        cv_summary["Model"] == "Features6_Leaf2_MapMatched"
    ].iloc[0]

    manifest_df = pd.DataFrame(manifest)
    manifest_df["CV_OA"] = float(map_matched["Pooled_OA"])
    manifest_df["CV_Kappa"] = float(map_matched["Pooled_Kappa"])
    manifest_df["CV_Balanced_Accuracy"] = float(map_matched["Balanced_Accuracy"])
    manifest_df["CV_Macro_F1"] = float(map_matched["Macro_F1"])
    manifest_df["GEE_Training_OA"] = float(gee_summary.iloc[0]["Training_OA"])
    manifest_df["GEE_Approx_OOB_Accuracy"] = float(
        gee_summary.iloc[0]["Approx_OOB_Accuracy"]
    )
    manifest_df.to_csv(LOCAL_OUTPUT_DIR / "S2_Annual_Map_Manifest.csv", index=False)

    settings = {
        "ee_project": EE_PROJECT,
        "roi_asset": ROI_ASSET,
        "training_points_asset": TRAINING_POINTS_ASSET,
        "sample_asset": RF_SAMPLE_ASSET,
        "terrain_asset": TERRAIN_ASSET,
        "classifier_asset": CLASSIFIER_ASSET,
        "habitat_years": HABITAT_YEARS,
        "training_habitat_year": TRAINING_HABITAT_YEAR,
        "predictor_bands": PREDICTOR_BANDS,
        "cloud_score_threshold": CLEAR_THRESHOLD,
        "trees": GEE_RF_TREES,
        "variables_per_split": MAX_FEATURES,
        "min_leaf_population": MIN_SAMPLES_LEAF,
        "bag_fraction": GEE_BAG_FRACTION,
        "random_seed": RANDOM_SEED,
        "export_crs": EXPORT_CRS,
        "export_transform": EXPORT_TRANSFORM,
        "class_map": CLASS_MAP,
        "class_palette": CLASS_PALETTE,
    }
    with (LOCAL_OUTPUT_DIR / "S2_Run_Configuration.json").open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    initialize_ee()
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    roi_fc = ee.FeatureCollection(ROI_ASSET)
    roi = roi_fc.geometry()
    terrain = build_terrain(roi)

    rf_samples = ensure_sample_asset(roi, terrain)
    samples_df = download_samples(rf_samples)

    cv_summary, _ = save_local_cv_results(samples_df)
    LOGGER.info("Five-fold CV summary:\n%s", cv_summary.round(3).to_string(index=False))

    gee_classifier = build_gee_classifier(rf_samples)
    gee_summary = save_gee_diagnostics(gee_classifier)
    LOGGER.info("GEE diagnostic summary:\n%s", gee_summary.round(3).to_string(index=False))

    deployed_classifier = ensure_classifier_asset(gee_classifier)

    asset_tasks, manifest = export_annual_assets(deployed_classifier, roi, terrain)
    if WAIT_FOR_MAP_ASSET_EXPORTS:
        wait_for_tasks(asset_tasks)

    drive_tasks = export_maps_to_drive(roi)
    if WAIT_FOR_DRIVE_EXPORTS:
        wait_for_tasks(drive_tasks, fail_on_error=False)

    save_run_metadata(cv_summary, gee_summary, manifest)

    LOGGER.info("Sentinel-2 workflow complete.")
    LOGGER.info("Local tables: %s", LOCAL_OUTPUT_DIR)
    if drive_tasks and not WAIT_FOR_DRIVE_EXPORTS:
        LOGGER.info("Google Drive tasks were launched and continue on the EE server.")


if __name__ == "__main__":
    main()
