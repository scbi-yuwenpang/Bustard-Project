#!/usr/bin/env python
"""
Preliminary comparison of Sentinel-2 temporal windows for Eastern Morocco.

Windows:
  Full_HY : 2024-09-01 to 2025-09-01
  Mar_Aug : 2025-03-01 to 2025-09-01
  Apr_Aug : 2025-04-01 to 2025-09-01
  May_Aug : 2025-05-01 to 2025-09-01

This is a screening test only:
- reuse existing RF points;
- keep one existing point per source polygon (~3,435 points);
- materialize each sampled table with a GEE batch export;
- download the stored table locally;
- use one fixed holdout (cv_fold == 0) only;
- NO 5-fold cross-validation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

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
# 0. SETTINGS
# =============================================================================

EE_PROJECT = "rse-global-wetlands"
ASSET_ROOT = "projects/rse-global-wetlands/assets"

ROI_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_Working_Area"
RF_POINTS_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_RF_Points_HY2025"
TERRAIN_ASSET = f"{ASSET_ROOT}/Eastern_Morocco_Terrain_GLO30"

LOCAL_ROOT = Path(
    r"C:\Users\PangY\OneDrive - Smithsonian Institution\Bustard"
    r"\01_Data\Habitat mapping\Temporal_Window_Preliminary"
)
SAMPLE_DIR = LOCAL_ROOT / "Samples"
ACCURACY_DIR = LOCAL_ROOT / "Accuracy"

WINDOWS: Dict[str, Tuple[str, str]] = {
    "Full_HY": ("2024-09-01", "2025-09-01"),
    "Mar_Aug": ("2025-03-01", "2025-09-01"),
    "Apr_Aug": ("2025-04-01", "2025-09-01"),
    "May_Aug": ("2025-05-01", "2025-09-01"),
}

WINDOW_TAG = {
    "Full_HY": "FullHY",
    "Mar_Aug": "MarAug",
    "Apr_Aug": "AprAug",
    "May_Aug": "MayAug",
}

# Safe reruns
REBUILD_SAMPLE_ASSETS = False
REBUILD_LOCAL_CSVS = False

# GEE
POLL_SECONDS = 30
SAMPLE_SCALE = 10
SAMPLE_TILE_SCALE = 16

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CLOUD_SCORE_BAND = "cs_cdf"
CLEAR_THRESHOLD = 0.60
MAX_SCENE_CLOUD_PERCENT = 10

# Predictors
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
INDEX_BANDS = ["EVI", "MSAVI", "NDWI", "BSI"]
TERRAIN_BANDS = ["Elevation", "Slope", "Northness", "Eastness"]
PREDICTOR_BANDS = S2_BANDS + INDEX_BANDS + TERRAIN_BANDS

# Sample fields
SAMPLE_ID_FIELD = "sample_id"
FEATURE_ID_FIELD = "feature_id"
CLASS_FIELD = "class_id"
GROUP_FIELD = "cv_fold"
SAMPLE_PROPERTIES = [
    SAMPLE_ID_FIELD,
    FEATURE_ID_FIELD,
    CLASS_FIELD,
    GROUP_FIELD,
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

# One fixed holdout only; no fold rotation.
VALIDATION_GROUP = 0

# Lightweight RF for screening
RF_TREES = 200
RF_MAX_FEATURES = 6
RF_MIN_LEAF = 2
RANDOM_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("temporal_window_preliminary")

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "CANCEL_REQUESTED"}


# =============================================================================
# 1. EARTH ENGINE HELPERS
# =============================================================================

def initialize_ee() -> None:
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        LOGGER.info("Earth Engine authentication required.")
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)
    LOGGER.info("Earth Engine initialized.")


def asset_exists(asset_id: str) -> bool:
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def wait_for_task(task: ee.batch.Task, label: str) -> None:
    previous = None
    while True:
        status = task.status()
        state = str(status.get("state", "UNKNOWN"))

        if state != previous:
            LOGGER.info("%s | %s", label, state)
            previous = state

        if state in TERMINAL_STATES:
            if state != "COMPLETED":
                raise RuntimeError(
                    f"Earth Engine task failed:\n{label}\n"
                    f"{status.get('error_message', '')}"
                )
            return

        time.sleep(POLL_SECONDS)


def sample_asset_id(window_name: str) -> str:
    return (
        f"{ASSET_ROOT}/Eastern_Morocco_TemporalWindow_Prelim_"
        f"HY2025_{WINDOW_TAG[window_name]}_1PtPerPolygon"
    )


# =============================================================================
# 2. PREFLIGHT
# =============================================================================

def preflight() -> None:
    required = [ROI_ASSET, RF_POINTS_ASSET, TERRAIN_ASSET]
    missing = [x for x in required if not asset_exists(x)]
    if missing:
        raise FileNotFoundError("Missing GEE assets:\n  " + "\n  ".join(missing))

    points = ee.FeatureCollection(RF_POINTS_ASSET)
    n_points = int(points.size().getInfo())
    n_polygons = int(points.aggregate_count_distinct(FEATURE_ID_FIELD).getInfo())
    props = ee.Feature(points.first()).propertyNames().getInfo()
    groups = sorted(points.aggregate_array(GROUP_FIELD).distinct().getInfo())

    missing_props = [x for x in SAMPLE_PROPERTIES if x not in props]
    if missing_props:
        raise KeyError(f"RF point asset missing fields: {missing_props}")

    LOGGER.info("Fixed points: %d", n_points)
    LOGGER.info("Source polygons: %d", n_polygons)
    LOGGER.info("Existing group labels: %s", groups)

    if VALIDATION_GROUP not in groups:
        raise ValueError(
            f"Validation group {VALIDATION_GROUP} not found; available={groups}"
        )

    terrain_bands = ee.Image(TERRAIN_ASSET).bandNames().getInfo()
    missing_terrain = [x for x in TERRAIN_BANDS if x not in terrain_bands]
    if missing_terrain:
        raise KeyError(f"Terrain asset missing bands: {missing_terrain}")

    LOGGER.info("Preflight QA passed.")


# =============================================================================
# 3. SENTINEL-2 PROCESSING
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
        "2.5*(NIR-RED)/(NIR+6*RED-7.5*BLUE+1)",
        {"NIR": nir, "RED": red, "BLUE": blue},
    ).rename("EVI")

    term = nir.multiply(2).add(1)
    disc = term.pow(2).subtract(nir.subtract(red).multiply(8)).max(0)
    msavi = term.subtract(disc.sqrt()).divide(2).rename("MSAVI")

    ndwi = image.expression(
        "(GREEN-NIR)/(GREEN+NIR)",
        {"GREEN": green, "NIR": nir},
    ).rename("NDWI")

    bsi = image.expression(
        "((SWIR+RED)-(NIR+BLUE))/((SWIR+RED)+(NIR+BLUE))",
        {"SWIR": swir1, "RED": red, "NIR": nir, "BLUE": blue},
    ).rename("BSI")

    return image.addBands([evi, msavi, ndwi, bsi])


def build_predictor_stack(
    start_date: str,
    end_date: str,
    roi: ee.Geometry,
    terrain: ee.Image,
) -> Tuple[ee.Image, ee.Image]:

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

    clean = (
        s2.linkCollection(cloud_score, [CLOUD_SCORE_BAND])
        .map(prepare_s2)
    )

    # Same definition for every candidate:
    # median reflectance first, then calculate indices.
    composite = clean.select(S2_BANDS).median()
    spectral = add_indices(composite)

    valid_obs = (
        clean.select("B2")
        .count()
        .rename("valid_obs")
        .unmask(0)
        .toUint16()
    )

    stack = (
        spectral.select(S2_BANDS + INDEX_BANDS)
        .addBands(terrain)
        .select(PREDICTOR_BANDS)
        .float()
    )

    return stack, valid_obs


# =============================================================================
# 4. ONE EXISTING POINT PER POLYGON
# =============================================================================

def get_preliminary_points() -> ee.FeatureCollection:
    all_points = ee.FeatureCollection(RF_POINTS_ASSET)

    # No new random sampling: retain one existing point per source polygon.
    points = (
        all_points
        .sort(SAMPLE_ID_FIELD)
        .distinct([FEATURE_ID_FIELD])
    )

    n = int(points.size().getInfo())
    LOGGER.info("Preliminary reference points: %d", n)

    if n != 3435:
        LOGGER.warning("Expected 3,435 points; found %d.", n)

    return points


# =============================================================================
# 5. MATERIALIZE SAMPLE TABLES WITH BATCH EXPORTS
# =============================================================================

def ensure_sample_asset(
    window_name: str,
    points: ee.FeatureCollection,
    roi: ee.Geometry,
    terrain: ee.Image,
) -> ee.FeatureCollection:

    asset_id = sample_asset_id(window_name)

    if asset_exists(asset_id) and not REBUILD_SAMPLE_ASSETS:
        LOGGER.info("%s | reusing GEE sample asset.", window_name)
        return ee.FeatureCollection(asset_id)

    if asset_exists(asset_id):
        LOGGER.warning("%s | deleting old sample asset.", window_name)
        ee.data.deleteAsset(asset_id)

    start_date, end_date = WINDOWS[window_name]

    LOGGER.info(
        "%s | building predictors %s -> %s",
        window_name,
        start_date,
        end_date,
    )

    stack, valid_obs = build_predictor_stack(
        start_date,
        end_date,
        roi,
        terrain,
    )

    extraction_image = stack.addBands(valid_obs)

    # geometries=True is important for Export.table.toAsset().
    samples = (
        extraction_image.sampleRegions(
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
                    "valid_obs",
                    SAMPLE_ID_FIELD,
                    FEATURE_ID_FIELD,
                    CLASS_FIELD,
                    GROUP_FIELD,
                ]
            )
        )
    )

    def add_meta(feature):
        return ee.Feature(feature).set(
            {
                "temporal_window": window_name,
                "start_date": start_date,
                "end_date_exclusive": end_date,
            }
        )

    samples = samples.map(add_meta)

    task = ee.batch.Export.table.toAsset(
        collection=samples,
        description=f"TemporalWindow_Prelim_{WINDOW_TAG[window_name]}_HY2025",
        assetId=asset_id,
    )

    task.start()

    LOGGER.info(
        "%s | started batch sample export: %s",
        window_name,
        task.id,
    )

    wait_for_task(task, f"{window_name} sample table")

    return ee.FeatureCollection(asset_id)


# =============================================================================
# 6. DOWNLOAD STORED TABLES
# =============================================================================

def normalize_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype("Int64").astype(str)
    return series.astype(str).str.strip()


def clean_dataframe(df: pd.DataFrame, window_name: str) -> pd.DataFrame:
    required = (
        PREDICTOR_BANDS
        + [
            "valid_obs",
            SAMPLE_ID_FIELD,
            FEATURE_ID_FIELD,
            CLASS_FIELD,
            GROUP_FIELD,
        ]
    )

    missing = [x for x in required if x not in df.columns]
    if missing:
        raise KeyError(f"{window_name}: missing columns {missing}")

    out = df.copy()

    out[SAMPLE_ID_FIELD] = normalize_id(out[SAMPLE_ID_FIELD])
    out[FEATURE_ID_FIELD] = out[FEATURE_ID_FIELD].astype(str).str.strip()

    for col in PREDICTOR_BANDS + ["valid_obs", CLASS_FIELD, GROUP_FIELD]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=required).copy()
    out[CLASS_FIELD] = out[CLASS_FIELD].astype(int)
    out[GROUP_FIELD] = out[GROUP_FIELD].astype(int)

    if out[SAMPLE_ID_FIELD].duplicated().any():
        raise ValueError(f"{window_name}: duplicate sample_id values found.")

    LOGGER.info(
        "%s | valid local rows=%d; polygons=%d",
        window_name,
        len(out),
        out[FEATURE_ID_FIELD].nunique(),
    )

    return out


def load_sample_table(
    window_name: str,
    sample_asset: ee.FeatureCollection,
) -> pd.DataFrame:

    local_csv = SAMPLE_DIR / f"Samples_{window_name}_HY2025.csv"

    if local_csv.exists() and not REBUILD_LOCAL_CSVS:
        LOGGER.info("%s | loading local CSV.", window_name)
        return clean_dataframe(pd.read_csv(local_csv), window_name)

    keep = (
        SAMPLE_PROPERTIES
        + PREDICTOR_BANDS
        + [
            "valid_obs",
            "temporal_window",
            "start_date",
            "end_date_exclusive",
        ]
    )

    # This is now cheap: the Sentinel-2 calculation is already materialized.
    LOGGER.info("%s | downloading materialized table.", window_name)

    df = geemap.ee_to_df(
        sample_asset.select(keep, retainGeometry=False)
    )

    df.to_csv(local_csv, index=False)
    LOGGER.info("%s | saved local CSV: %s", window_name, local_csv)

    return clean_dataframe(df, window_name)


# =============================================================================
# 7. SAME SAMPLE IDS FOR ALL WINDOWS
# =============================================================================

def common_samples(
    window_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:

    common_ids = set.intersection(
        *[
            set(df[SAMPLE_ID_FIELD].tolist())
            for df in window_dfs.values()
        ]
    )

    if not common_ids:
        raise RuntimeError("No common sample IDs across temporal windows.")

    LOGGER.info("Common samples across all windows: %d", len(common_ids))

    out: Dict[str, pd.DataFrame] = {}

    for name, df in window_dfs.items():
        out[name] = (
            df[df[SAMPLE_ID_FIELD].isin(common_ids)]
            .copy()
            .sort_values(SAMPLE_ID_FIELD)
            .reset_index(drop=True)
        )

    # Verify class, polygon, and holdout group are identical.
    ref_name = next(iter(out))
    ref = (
        out[ref_name][
            [SAMPLE_ID_FIELD, FEATURE_ID_FIELD, CLASS_FIELD, GROUP_FIELD]
        ]
        .set_index(SAMPLE_ID_FIELD)
        .sort_index()
    )

    for name, df in out.items():
        check = (
            df[
                [SAMPLE_ID_FIELD, FEATURE_ID_FIELD, CLASS_FIELD, GROUP_FIELD]
            ]
            .set_index(SAMPLE_ID_FIELD)
            .sort_index()
        )

        if not ref.equals(check):
            raise ValueError(f"Metadata mismatch: {ref_name} vs {name}")

    return out


# =============================================================================
# 8. ONE FIXED HOLDOUT — NO CROSS-VALIDATION
# =============================================================================

def evaluate_window(
    df: pd.DataFrame,
    window_name: str,
) -> Dict[str, Any]:

    train = df[df[GROUP_FIELD] != VALIDATION_GROUP].copy()
    valid = df[df[GROUP_FIELD] == VALIDATION_GROUP].copy()

    if train.empty or valid.empty:
        raise ValueError(f"{window_name}: empty train/validation subset.")

    LOGGER.info(
        "%s | training=%d | validation=%d",
        window_name,
        len(train),
        len(valid),
    )

    model = RandomForestClassifier(
        n_estimators=RF_TREES,
        max_features=RF_MAX_FEATURES,
        min_samples_leaf=RF_MIN_LEAF,
        class_weight=None,
        bootstrap=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(
        train[PREDICTOR_BANDS],
        train[CLASS_FIELD],
    )

    pred = model.predict(valid[PREDICTOR_BANDS]).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        valid[CLASS_FIELD],
        pred,
        labels=CLASS_IDS,
        zero_division=0,
    )

    overall = {
        "Window": window_name,
        "N_total": len(df),
        "N_training": len(train),
        "N_validation": len(valid),
        "OA": accuracy_score(valid[CLASS_FIELD], pred),
        "Kappa": cohen_kappa_score(valid[CLASS_FIELD], pred),
        "Balanced_Accuracy": balanced_accuracy_score(valid[CLASS_FIELD], pred),
        "Macro_F1": float(np.mean(f1)),
        "ValidObs_mean": float(df["valid_obs"].mean()),
        "ValidObs_p10": float(df["valid_obs"].quantile(0.10)),
        "ValidObs_median": float(df["valid_obs"].median()),
        "Points_GE5obs_pct": float(100 * (df["valid_obs"] >= 5).mean()),
        "Points_GE10obs_pct": float(100 * (df["valid_obs"] >= 10).mean()),
    }

    per_class = pd.DataFrame(
        {
            "Window": window_name,
            "class_id": CLASS_IDS,
            "Habitat": [CLASS_MAP[i] for i in CLASS_IDS],
            "N_validation": support,
            "Precision_UserAcc": precision,
            "Recall_ProducerAcc": recall,
            "F1": f1,
        }
    )

    cm = confusion_matrix(
        valid[CLASS_FIELD],
        pred,
        labels=CLASS_IDS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[CLASS_MAP[i] for i in CLASS_IDS],
        columns=[CLASS_MAP[i] for i in CLASS_IDS],
    )
    cm_df.index.name = "Actual"
    cm_df.columns.name = "Predicted"

    low_support = per_class[per_class["N_validation"] < 5]
    if not low_support.empty:
        LOGGER.warning(
            "%s | classes with <5 holdout samples: %s",
            window_name,
            ", ".join(
                (
                    low_support["Habitat"]
                    + " (n="
                    + low_support["N_validation"].astype(str)
                    + ")"
                ).tolist()
            ),
        )

    return {
        "overall": overall,
        "per_class": per_class,
        "confusion": cm_df,
    }


# =============================================================================
# 9. SAVE RESULTS
# =============================================================================

def save_results(results: Dict[str, Dict[str, Any]]) -> None:

    overall = pd.DataFrame(
        [x["overall"] for x in results.values()]
    ).sort_values(
        ["Balanced_Accuracy", "Macro_F1", "Kappa"],
        ascending=False,
    )

    per_class = pd.concat(
        [x["per_class"] for x in results.values()],
        ignore_index=True,
    )

    recall_wide = (
        per_class.pivot(
            index=["class_id", "Habitat", "N_validation"],
            columns="Window",
            values="Recall_ProducerAcc",
        )
        .reset_index()
    )

    f1_wide = (
        per_class.pivot(
            index=["class_id", "Habitat", "N_validation"],
            columns="Window",
            values="F1",
        )
        .reset_index()
    )

    overall.to_csv(
        ACCURACY_DIR / "Temporal_Window_Preliminary_Overall.csv",
        index=False,
    )
    per_class.to_csv(
        ACCURACY_DIR / "Temporal_Window_Preliminary_PerClass.csv",
        index=False,
    )
    recall_wide.to_csv(
        ACCURACY_DIR / "Temporal_Window_Preliminary_Recall.csv",
        index=False,
    )
    f1_wide.to_csv(
        ACCURACY_DIR / "Temporal_Window_Preliminary_F1.csv",
        index=False,
    )

    for name, result in results.items():
        result["confusion"].to_csv(
            ACCURACY_DIR / f"Temporal_Window_Preliminary_CM_{name}.csv"
        )

    workbook = ACCURACY_DIR / "Temporal_Window_Preliminary_HY2025.xlsx"
    try:
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            overall.to_excel(writer, sheet_name="Overall", index=False)
            per_class.to_excel(writer, sheet_name="Per_class", index=False)
            recall_wide.to_excel(writer, sheet_name="Recall", index=False)
            f1_wide.to_excel(writer, sheet_name="F1", index=False)
    except Exception as exc:
        LOGGER.warning("Excel export failed: %s", exc)

    print("\n" + "=" * 105)
    print("PRELIMINARY TEMPORAL-WINDOW COMPARISON")
    print("ONE FIXED HOLDOUT ONLY — NO CROSS-VALIDATION")
    print("=" * 105)

    cols = [
        "Window",
        "N_training",
        "N_validation",
        "OA",
        "Kappa",
        "Balanced_Accuracy",
        "Macro_F1",
        "ValidObs_p10",
        "ValidObs_median",
        "Points_GE5obs_pct",
        "Points_GE10obs_pct",
    ]

    print(overall[cols].round(3).to_string(index=False))

    print("\nPER-CLASS RECALL / PRODUCER ACCURACY")
    print(recall_wide.round(3).to_string(index=False))

    print("\nPER-CLASS F1")
    print(f1_wide.round(3).to_string(index=False))

    print("\nOutputs:")
    print(ACCURACY_DIR)


# =============================================================================
# 10. MAIN
# =============================================================================

def main() -> None:

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    ACCURACY_DIR.mkdir(parents=True, exist_ok=True)

    initialize_ee()
    preflight()

    roi = ee.FeatureCollection(ROI_ASSET).geometry()

    terrain = (
        ee.Image(TERRAIN_ASSET)
        .select(TERRAIN_BANDS)
        .resample("bilinear")
    )

    points = get_preliminary_points()

    # Step 1: materialize/reuse four small GEE sample tables.
    assets: Dict[str, ee.FeatureCollection] = {}

    for name in WINDOWS:
        LOGGER.info("=" * 72)
        LOGGER.info(
            "Preparing %s | %s -> %s",
            name,
            WINDOWS[name][0],
            WINDOWS[name][1],
        )

        assets[name] = ensure_sample_asset(
            name,
            points,
            roi,
            terrain,
        )

    # Step 2: download only materialized tables.
    dfs: Dict[str, pd.DataFrame] = {}

    for name in WINDOWS:
        dfs[name] = load_sample_table(
            name,
            assets[name],
        )

    # Step 3: exact same reference samples for all windows.
    dfs = common_samples(dfs)

    # Step 4: one fixed holdout only.
    results: Dict[str, Dict[str, Any]] = {}

    for name in WINDOWS:
        LOGGER.info("=" * 72)
        LOGGER.info("Preliminary accuracy test: %s", name)

        results[name] = evaluate_window(
            dfs[name],
            name,
        )

    save_results(results)

    LOGGER.info("Preliminary temporal-window screening finished.")


if __name__ == "__main__":
    main()
