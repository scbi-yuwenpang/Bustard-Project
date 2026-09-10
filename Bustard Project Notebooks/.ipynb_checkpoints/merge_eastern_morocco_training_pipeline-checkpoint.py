#!/usr/bin/env python
"""
eastern_morocco_training_pipeline.py

Build the pilot/master habitat-training dataset for Eastern Morocco from the
currently prepared reference sources.

Designed for BOTH:
1. import from Jupyter; and
2. command-line execution.

Jupyter example
---------------
import sys
sys.path.append("C:/path/to/folder/containing/this/script")

import eastern_morocco_training_pipeline as emt

cfg = emt.PipelineConfig()
result = emt.run_pipeline(
    cfg,
    conflict_policy="flag_only",  # keep conflicts for first pilot, but flag them
    export=True,
    make_plots=True,
)

training_pilot = result["training_pilot"]
class_summary = result["class_summary"]
source_summary = result["source_summary"]
conflicts = result["conflicts"]

display(class_summary)
display(source_summary)

Methodological notes
--------------------
- Only polygon training candidates are merged into the master training layer.
- HOT OSM waterway lines are retained as an ANCILLARY reference layer only.
- If the Trophic or Hayat source is still a point layer, this pilot workflow
  can buffer those points by 10 m. Such geometry is explicitly tagged as a
  pilot point-buffer rule and can be replaced later by VHR-refined polygons.
- Al Baten areal and Wadi candidates are expected to be outputs from the
  dedicated Al Baten preprocessing scripts.
- Cross-class overlaps are detected and flagged. By default, they are NOT
  automatically removed for the first pilot classification.
- Class/pixel balancing is NOT done here. It should be applied later when
  extracting Sentinel-2 training pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# FINAL HABITAT LEGEND
# =====================================================================

CLASS_DICT: Dict[str, int] = {
    "Grass steppe": 0,
    "Shrub steppe": 1,
    "Wooded steppe": 2,
    "Bare rocky": 3,
    "Spreading area": 4,
    "Salty steppe": 5,
    "Dune": 6,
    "Wadi and gullies": 7,
    "Cultivated field/fallow": 8,
    "Water body": 9,
    "Built-up": 10,
}


CLASS_CROSSWALK: Dict[str, str] = {
    # Final labels
    "Grass steppe": "Grass steppe",
    "Shrub steppe": "Shrub steppe",
    "Wooded steppe": "Wooded steppe",
    "Bare rocky": "Bare rocky",
    "Spreading area": "Spreading area",
    "Salty steppe": "Salty steppe",
    "Dune": "Dune",
    "Wadi and gullies": "Wadi and gullies",
    "Cultivated field/fallow": "Cultivated field/fallow",
    "Water body": "Water body",
    "Built-up": "Built-up",

    # Historical / field variants
    "Open woodland": "Wooded steppe",
    "Reg": "Bare rocky",
    "Wadi & gullies": "Wadi and gullies",
    "Wadi and gully": "Wadi and gullies",

    # Trophic variants
    "Alfah-grass steppe": "Grass steppe",
    "Alfa-grass steppe": "Grass steppe",
    "Chamaephyte steppe": "Shrub steppe",
    "Anabasis steppe": "Shrub steppe",

    # Al Baten / interpreted-map variants
    "Alluvial silt plain": "Spreading area",
    "Tree farming": "Cultivated field/fallow",
    "Wadi": "Wadi and gullies",
    "Flooded wadi": "Wadi and gullies",
    "Building": "Built-up",
}


MASTER_FIELDS = [
    "sample_id",
    "site_id",
    "source_id",
    "class_raw",
    "class",
    "class_id",
    "source",
    "ref_year",
    "field_sup",
    "confidence",
    "geom_rule",
    "grid_id",
    "qc_status",
    "use_status",
    "poly_area",
    "geometry",
]


# =====================================================================
# PATH CONFIGURATION
# =====================================================================

@dataclass
class PipelineConfig:
    """
    Default paths reflect the current Eastern Morocco project structure.

    Edit these once if your local filenames/folders differ. The same script
    can then be imported repeatedly without rerunning notebook preprocessing.
    """

    base: Path = Path(
        r"C:\Users\PangY\OneDrive - Smithsonian Institution\Bustard\01_Data"
    )

    # ---------------------- Existing field footprints -----------------
    phyto_path: Optional[Path] = None
    phyto_layer: str = "field_footprints"

    # ---------------------- Trophic survey -----------------------------
    # May be a .shp/.gpkg file OR a folder/base name.
    trophic_path: Optional[Path] = None
    trophic_layer: Optional[str] = None
    trophic_class_field: Optional[str] = "class"
    trophic_point_buffer_m: float = 10.0

    # ---------------------- Hayat field source -------------------------
    # Optional. If the path can be resolved, it is included.
    include_hayat: bool = True
    hayat_path: Optional[Path] = None
    hayat_layer: Optional[str] = None
    hayat_class_field: Optional[str] = "class"
    hayat_point_buffer_m: float = 10.0

    # ---------------------- Anthropization -----------------------------
    albaten_farms_path: Optional[Path] = None
    bouarfa_2024_path: Optional[Path] = None

    # ---------------------- Built-up -----------------------------------
    builtup_path: Optional[Path] = None
    builtup_layer: str = "builtup_selected"

    # ---------------------- Al Baten prepared candidates ---------------
    albaten_areal_path: Optional[Path] = None
    albaten_areal_layer: str = "training_patches"

    albaten_wadi_path: Optional[Path] = None
    albaten_wadi_layer: str = "draft_wadi_segments"

    # ---------------------- Regional grid ------------------------------
    grid_path: Optional[Path] = None
    grid_layer: str = "grid_20km"

    # ---------------------- Ancillary drainage reference ---------------
    waterways_path: Optional[Path] = None

    # ---------------------- Master output ------------------------------
    output_path: Optional[Path] = None

    # ---------------------- Conflict threshold -------------------------
    min_conflict_overlap_m2: float = 25.0

    def __post_init__(self):
        train_dir = (
            self.base
            / r"Reneco\Vegetation Cover and Food Availability Studies"
        )
        anthro_dir = (
            self.base
            / r"Reneco\GIS Data\Extract data Yuwen\Shapefiles\anthropisation"
        )
        habitat_dir = self.base / r"Habitat mapping"

        if self.phyto_path is None:
            self.phyto_path = (
                train_dir / "Eastern_Morocco_Field_Footprints.gpkg"
            )

        if self.trophic_path is None:
            self.trophic_path = (
                train_dir
                / "Trophic_survey_data_114_station_habitat_26_Eastern_Morocco"
            )

        if self.hayat_path is None:
            self.hayat_path = (
                self.base
                / r"Reneco\GIS Data\Extract data Yuwen\Shapefiles"
                / r"Habitat Mapping Shapefiles"
                / "Phytosociologicalsurveys_fromHayat_edited"
            )

        if self.albaten_farms_path is None:
            self.albaten_farms_path = (
                anthro_dir / "Al_Baten_occupation_Farms_20240621.shp"
            )

        if self.bouarfa_2024_path is None:
            self.bouarfa_2024_path = (
                anthro_dir / "Bouarfa_anthropized_land_1_2024.shp"
            )

        if self.builtup_path is None:
            self.builtup_path = (
                train_dir / "Eastern_Morocco_BuiltUp_Candidates_2025.gpkg"
            )

        if self.albaten_areal_path is None:
            self.albaten_areal_path = (
                habitat_dir / "Al_Baten_Training_2025.gpkg"
            )

        if self.albaten_wadi_path is None:
            self.albaten_wadi_path = (
                habitat_dir / "Al_Baten_Wadi_Training_2025.gpkg"
            )

        if self.grid_path is None:
            self.grid_path = (
                train_dir / "Eastern_Morocco_Training_Grid.gpkg"
            )

        if self.waterways_path is None:
            self.waterways_path = (
                habitat_dir
                / "waterbody"
                / "hotosm_mar_waterways_lines_EM.shp"
            )

        if self.output_path is None:
            self.output_path = (
                train_dir / "Eastern_Morocco_Training_Master_2025.gpkg"
            )


# =====================================================================
# I/O HELPERS
# =====================================================================

def resolve_vector_path(path: Path, required: bool = True) -> Optional[Path]:
    """
    Resolve a vector source supplied as:
    - exact .shp/.gpkg/.geojson path,
    - a base filename without extension, or
    - a directory containing one vector file.
    """
    path = Path(path)

    if path.is_file():
        return path

    for ext in [".shp", ".gpkg", ".geojson"]:
        candidate = Path(str(path) + ext)
        if candidate.is_file():
            return candidate

    if path.is_dir():
        candidates: List[Path] = []
        for pattern in ["*.shp", "*.gpkg", "*.geojson"]:
            candidates.extend(path.glob(pattern))

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            names = "\n  - ".join(str(p) for p in candidates)
            raise ValueError(
                f"Multiple vector files found under:\n{path}\n"
                f"Specify one explicitly:\n  - {names}"
            )

    if required:
        raise FileNotFoundError(f"Could not locate vector source: {path}")

    return None


def read_vector(
    path: Path,
    layer: Optional[str] = None,
    required: bool = True,
) -> Optional[gpd.GeoDataFrame]:
    resolved = resolve_vector_path(path, required=required)
    if resolved is None:
        return None

    if resolved.suffix.lower() == ".gpkg" and layer:
        return gpd.read_file(resolved, layer=layer)

    return gpd.read_file(resolved)


# =====================================================================
# GEOMETRY + CLASS HELPERS
# =====================================================================

def normalize_class_values(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace(CLASS_CROSSWALK)
    )


def clean_geometries(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    out = gdf.to_crs(target_crs).copy()

    keep = out.geometry.apply(
        lambda g: g is not None and not g.is_empty
    )
    out = out[keep].copy()

    invalid = ~out.geometry.is_valid
    if invalid.any():
        out.loc[invalid, "geometry"] = (
            out.loc[invalid, "geometry"].buffer(0)
        )

    keep = out.geometry.apply(
        lambda g: g is not None and not g.is_empty and g.is_valid
    )

    return out[keep].copy()


def ensure_polygon_geometry(
    gdf: gpd.GeoDataFrame,
    target_crs,
    point_buffer_m: Optional[float] = None,
) -> Tuple[gpd.GeoDataFrame, str]:
    """
    Ensure a model-ready polygon layer.

    If a source consists of points and point_buffer_m is given, point
    features are buffered for the PILOT workflow.
    """
    out = clean_geometries(gdf, target_crs)
    geom_types = set(out.geometry.geom_type.unique())

    polygon_types = {"Polygon", "MultiPolygon"}
    point_types = {"Point", "MultiPoint"}

    if geom_types.issubset(polygon_types):
        return out, "SOURCE_POLYGON"

    if geom_types.issubset(point_types):
        if point_buffer_m is None:
            raise ValueError(
                "Point geometry found but no point_buffer_m was provided."
            )

        # MultiPoint is uncommon here; explode before buffering.
        out = (
            out.explode(index_parts=False)
            .reset_index(drop=True)
        )
        out["geometry"] = out.geometry.buffer(point_buffer_m)

        return out, f"POINT_BUFFER_{int(point_buffer_m)}M_PILOT"

    raise ValueError(
        f"Mixed/unsupported geometry types encountered: {sorted(geom_types)}"
    )


def assign_grid(
    gdf: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    out = gdf.reset_index(drop=True).copy()

    pts = gpd.GeoDataFrame(
        {"_row_id": np.arange(len(out))},
        geometry=out.geometry.representative_point(),
        crs=out.crs,
    )

    joined = gpd.sjoin(
        pts,
        grid[["grid_id", "geometry"]],
        how="left",
        predicate="within",
    )

    joined = (
        joined.sort_values("_row_id")
        .drop_duplicates("_row_id", keep="first")
        .set_index("_row_id")
    )

    out["grid_id"] = (
        joined["grid_id"]
        .reindex(np.arange(len(out)))
        .values
    )

    return out


def auto_find_field(
    gdf: gpd.GeoDataFrame,
    preferred: Optional[str],
    candidates: List[str],
) -> Optional[str]:
    if preferred and preferred in gdf.columns:
        return preferred

    for field in candidates:
        if field in gdf.columns:
            return field

    return None


def standardize_training_source(
    gdf: gpd.GeoDataFrame,
    *,
    work_crs,
    grid: gpd.GeoDataFrame,
    prefix: str,
    source: str,
    class_field: Optional[str] = None,
    class_name: Optional[str] = None,
    source_id_field: Optional[str] = None,
    site_id_field: Optional[str] = None,
    ref_year: Optional[int] = None,
    field_sup: int = 0,
    confidence: int = 2,
    geom_rule: Optional[str] = None,
    point_buffer_m: Optional[float] = None,
) -> gpd.GeoDataFrame:

    out, inferred_geom_rule = ensure_polygon_geometry(
        gdf,
        target_crs=work_crs,
        point_buffer_m=point_buffer_m,
    )
    out = out.reset_index(drop=True)

    # --------------------------- class --------------------------------
    if class_name is not None:
        out["class_raw"] = class_name
        out["class"] = class_name
    else:
        field = auto_find_field(
            out,
            class_field,
            ["class", "Habitat", "habitat", "Class", "CLASS"],
        )
        if field is None:
            raise KeyError(
                f"{source}: could not find a class field. "
                f"Fields are {out.columns.tolist()}"
            )

        # Preserve an existing class_raw from preprocessing if present.
        if "class_raw" not in out.columns:
            out["class_raw"] = out[field].astype("string").str.strip()

        out["class"] = normalize_class_values(out[field])

    invalid_classes = sorted(
        set(
            out.loc[
                ~out["class"].isin(CLASS_DICT),
                "class",
            ]
            .dropna()
            .astype(str)
        )
    )

    if invalid_classes:
        raise ValueError(
            f"{source}: labels not mapped to the final legend:\n  - "
            + "\n  - ".join(invalid_classes)
        )

    out["class_id"] = out["class"].map(CLASS_DICT)

    # ---------------------- preserve source ID ------------------------
    sid_field = auto_find_field(
        out,
        source_id_field,
        [
            "source_id",
            "part_id",
            "wadi_part_id",
            "cluster_id",
            "Station_ID",
            "sample_id",
            "OBJECTID",
            "FID",
        ],
    )

    if sid_field:
        source_ids = out[sid_field].astype("string")
    else:
        source_ids = pd.Series(
            out.index.astype(str),
            index=out.index,
            dtype="string",
        )

    out["source_id"] = source_ids

    # ---------------------- preserve site ID --------------------------
    site_field = auto_find_field(
        out,
        site_id_field,
        ["site_id", "Station_ID", "part_id", "wadi_part_id", "cluster_id"],
    )

    if site_field:
        sites = out[site_field].astype("string")
    else:
        sites = out["source_id"].astype("string")

    # Prefix IDs so they are globally unique.
    out["sample_id"] = [
        f"{prefix}_{i:05d}" for i in range(1, len(out) + 1)
    ]
    out["site_id"] = prefix + "_" + sites

    # -------------------------- metadata ------------------------------
    out["source"] = source

    if "ref_year" not in out.columns:
        out["ref_year"] = ref_year
    else:
        out["ref_year"] = out["ref_year"].fillna(ref_year)

    if "field_sup" not in out.columns:
        out["field_sup"] = field_sup

    if "confidence" not in out.columns:
        out["confidence"] = confidence

    if "geom_rule" not in out.columns:
        out["geom_rule"] = (
            geom_rule if geom_rule is not None else inferred_geom_rule
        )
    else:
        out["geom_rule"] = out["geom_rule"].fillna(
            geom_rule if geom_rule is not None else inferred_geom_rule
        )

    if "qc_status" not in out.columns:
        out["qc_status"] = "UNREVIEWED"

    if "use_status" not in out.columns:
        out["use_status"] = "CANDIDATE"

    out["poly_area"] = out.geometry.area

    out = assign_grid(out, grid)

    return out


def final_schema(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    for col in MASTER_FIELDS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[MASTER_FIELDS].copy()


# =====================================================================
# CONFLICTS + SUMMARIES
# =====================================================================

def find_cross_class_conflicts(
    training: gpd.GeoDataFrame,
    min_overlap_m2: float = 25.0,
) -> Tuple[pd.DataFrame, set]:
    """
    Flag spatial overlaps between DIFFERENT habitat classes.
    Same-class overlaps are not treated as label conflicts.
    """
    sindex = training.sindex

    conflict_records: List[Dict[str, Any]] = []
    conflict_ids = set()

    for i, row_i in training.iterrows():
        possible = sindex.query(
            row_i.geometry,
            predicate="intersects",
        )

        for j in possible:
            if j <= i:
                continue

            row_j = training.iloc[j]

            if row_i["class"] == row_j["class"]:
                continue

            intersection = row_i.geometry.intersection(row_j.geometry)

            if intersection.is_empty:
                continue

            overlap_area = intersection.area

            if overlap_area < min_overlap_m2:
                continue

            area_i = row_i.geometry.area
            area_j = row_j.geometry.area

            conflict_records.append(
                {
                    "sample_1": row_i["sample_id"],
                    "class_1": row_i["class"],
                    "source_1": row_i["source"],
                    "sample_2": row_j["sample_id"],
                    "class_2": row_j["class"],
                    "source_2": row_j["source"],
                    "overlap_m2": overlap_area,
                    "overlap_pct_1": (
                        overlap_area / area_i * 100.0 if area_i > 0 else np.nan
                    ),
                    "overlap_pct_2": (
                        overlap_area / area_j * 100.0 if area_j > 0 else np.nan
                    ),
                }
            )

            conflict_ids.add(row_i["sample_id"])
            conflict_ids.add(row_j["sample_id"])

    conflicts = pd.DataFrame(conflict_records)

    if len(conflicts):
        conflicts = conflicts.sort_values(
            "overlap_m2",
            ascending=False,
        ).reset_index(drop=True)

    return conflicts, conflict_ids


def summarize_training(
    training: gpd.GeoDataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    source_summary = (
        training.groupby(["source", "class"])
        .agg(
            n_polygons=("sample_id", "size"),
            n_sites=("site_id", "nunique"),
            n_grids=("grid_id", "nunique"),
            total_area_m2=("poly_area", "sum"),
            median_area_m2=("poly_area", "median"),
        )
        .reset_index()
    )

    class_summary = (
        training.groupby("class")
        .agg(
            n_polygons=("sample_id", "size"),
            n_sites=("site_id", "nunique"),
            n_sources=("source", "nunique"),
            n_grids=("grid_id", "nunique"),
            total_area_m2=("poly_area", "sum"),
            median_area_m2=("poly_area", "median"),
        )
        .sort_values("n_sites", ascending=False)
    )

    return source_summary, class_summary


# =====================================================================
# PILOT SET POLICY
# =====================================================================

def make_pilot_set(
    training: gpd.GeoDataFrame,
    conflict_policy: str = "flag_only",
) -> gpd.GeoDataFrame:
    """
    conflict_policy:
        "flag_only"   -> keep everything for first pilot, but retain conflict_flag.
        "exclude_all" -> exclude every polygon involved in a cross-class conflict.

    Recommended for the first trial: "flag_only".
    """
    allowed = {"flag_only", "exclude_all"}

    if conflict_policy not in allowed:
        raise ValueError(
            f"conflict_policy must be one of {sorted(allowed)}"
        )

    if conflict_policy == "flag_only":
        pilot = training.copy()
    else:
        pilot = training[training["conflict_flag"] == 0].copy()

    pilot["pilot_use"] = 1

    return pilot.reset_index(drop=True)


# =====================================================================
# PLOTTING
# =====================================================================

def plot_training_maps(
    training: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    title_prefix: str = "Eastern Morocco",
):
    """
    Show:
    1. polygon candidate map;
    2. representative-site map.
    """

    fig, ax = plt.subplots(figsize=(15, 11))

    grid.boundary.plot(
        ax=ax,
        linewidth=0.35,
        alpha=0.30,
    )

    training.plot(
        ax=ax,
        column="class",
        categorical=True,
        cmap="tab20",
        alpha=0.60,
        edgecolor="black",
        linewidth=0.15,
        legend=True,
        legend_kwds={
            "title": "Habitat class",
            "bbox_to_anchor": (1.02, 1),
            "loc": "upper left",
        },
    )

    ax.set_title(
        f"{title_prefix} - Combined Training Candidate Polygons",
        fontsize=15,
    )
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    sites = training.copy()
    sites["geometry"] = sites.geometry.representative_point()

    fig, ax = plt.subplots(figsize=(15, 11))

    grid.boundary.plot(
        ax=ax,
        linewidth=0.35,
        alpha=0.30,
    )

    sites.plot(
        ax=ax,
        column="class",
        categorical=True,
        cmap="tab20",
        markersize=14,
        alpha=0.80,
        legend=True,
        legend_kwds={
            "title": "Habitat class",
            "bbox_to_anchor": (1.02, 1),
            "loc": "upper left",
        },
    )

    ax.set_title(
        f"{title_prefix} - Spatial Distribution of Training Sites",
        fontsize=15,
    )
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


# =====================================================================
# EXPORT
# =====================================================================

def export_pipeline_outputs(
    *,
    output_path: Path,
    training_candidates: gpd.GeoDataFrame,
    training_pilot: gpd.GeoDataFrame,
    field_grid: gpd.GeoDataFrame,
    source_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    conflicts: pd.DataFrame,
    conflict_ids: set,
    waterways: Optional[gpd.GeoDataFrame],
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        try:
            output_path.unlink()
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot overwrite {output_path}. "
                "Close it in ArcGIS Pro and rerun."
            ) from exc

    training_candidates.to_file(
        output_path,
        layer="training_candidates",
        driver="GPKG",
    )

    training_pilot.to_file(
        output_path,
        layer="training_pilot",
        driver="GPKG",
    )

    field_grid.to_file(
        output_path,
        layer="grid_20km",
        driver="GPKG",
    )

    if conflict_ids:
        conflict_features = training_candidates[
            training_candidates["sample_id"].isin(conflict_ids)
        ].copy()

        conflict_features.to_file(
            output_path,
            layer="conflict_review",
            driver="GPKG",
        )

    if waterways is not None and len(waterways):
        waterways.to_file(
            output_path,
            layer="ancillary_hot_waterways",
            driver="GPKG",
        )

    summary_base = output_path.with_suffix("")

    source_summary.to_csv(
        str(summary_base) + "_source_summary.csv",
        index=False,
    )

    class_summary.to_csv(
        str(summary_base) + "_class_summary.csv",
    )

    if len(conflicts):
        conflicts.to_csv(
            str(summary_base) + "_conflicts.csv",
            index=False,
        )


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def run_pipeline(
    cfg: Optional[PipelineConfig] = None,
    *,
    conflict_policy: str = "flag_only",
    export: bool = True,
    make_plots: bool = True,
) -> Dict[str, Any]:
    """
    Build the complete Eastern Morocco pilot training dataset.

    Returns a dictionary with:
        training_candidates
        training_pilot
        source_summary
        class_summary
        conflicts
        grid
        waterways
        source_counts
    """
    if cfg is None:
        cfg = PipelineConfig()

    print("=" * 72)
    print("EASTERN MOROCCO TRAINING PIPELINE")
    print("=" * 72)

    # --------------------------------------------------------------
    # 1. Regional grid / working CRS
    # --------------------------------------------------------------
    field_grid = read_vector(
        cfg.grid_path,
        layer=cfg.grid_layer,
        required=True,
    )

    if field_grid.crs is None:
        raise ValueError("Regional grid has no CRS.")

    work_crs = field_grid.crs

    print(f"Working CRS: {work_crs}")
    print(f"Grid cells:  {len(field_grid)}")

    sources: List[gpd.GeoDataFrame] = []
    source_counts: Dict[str, int] = {}

    # --------------------------------------------------------------
    # 2. Phytosociological footprints
    # --------------------------------------------------------------
    phyto_raw = read_vector(
        cfg.phyto_path,
        layer=cfg.phyto_layer,
        required=True,
    )

    phyto_std = standardize_training_source(
        phyto_raw,
        work_crs=work_crs,
        grid=field_grid,
        prefix="PHY",
        source="PHYTOSOC",
        class_field="class",
        source_id_field="Station_ID",
        site_id_field="Station_ID",
        field_sup=1,
        confidence=2,
        geom_rule="AREA_EQUIV_BUFFER",
    )

    sources.append(final_schema(phyto_std))
    source_counts["PHYTOSOC"] = len(phyto_std)

    # --------------------------------------------------------------
    # 3. Trophic survey
    # --------------------------------------------------------------
    trophic_raw = read_vector(
        cfg.trophic_path,
        layer=cfg.trophic_layer,
        required=True,
    )

    trophic_std = standardize_training_source(
        trophic_raw,
        work_crs=work_crs,
        grid=field_grid,
        prefix="TRO",
        source="TROPHIC_SURVEY",
        class_field=cfg.trophic_class_field,
        source_id_field="Station_ID",
        site_id_field="Station_ID",
        field_sup=1,
        confidence=3,
        geom_rule=None,
        point_buffer_m=cfg.trophic_point_buffer_m,
    )

    sources.append(final_schema(trophic_std))
    source_counts["TROPHIC_SURVEY"] = len(trophic_std)

    # --------------------------------------------------------------
    # 4. Hayat source, if available
    # --------------------------------------------------------------
    if cfg.include_hayat:
        hayat_raw = read_vector(
            cfg.hayat_path,
            layer=cfg.hayat_layer,
            required=False,
        )

        if hayat_raw is not None:
            hayat_std = standardize_training_source(
                hayat_raw,
                work_crs=work_crs,
                grid=field_grid,
                prefix="HAY",
                source="HAYAT_PHYTO",
                class_field=cfg.hayat_class_field,
                source_id_field="Station_ID",
                site_id_field="Station_ID",
                field_sup=1,
                confidence=2,
                geom_rule=None,
                point_buffer_m=cfg.hayat_point_buffer_m,
            )

            sources.append(final_schema(hayat_std))
            source_counts["HAYAT_PHYTO"] = len(hayat_std)
        else:
            print(
                "Note: Hayat source was not resolved; skipping optional source."
            )

    # --------------------------------------------------------------
    # 5. Al Baten farms - cultivated/fallow
    # --------------------------------------------------------------
    ab_farms_raw = read_vector(
        cfg.albaten_farms_path,
        required=True,
    )

    ab_farms_std = standardize_training_source(
        ab_farms_raw,
        work_crs=work_crs,
        grid=field_grid,
        prefix="ABF",
        source="ALBATEN_FARMS_2024",
        class_name="Cultivated field/fallow",
        ref_year=2024,
        field_sup=0,
        confidence=3,
        geom_rule="EXISTING_CULTIVATED_POLYGON",
    )

    sources.append(final_schema(ab_farms_std))
    source_counts["ALBATEN_FARMS_2024"] = len(ab_farms_std)

    # --------------------------------------------------------------
    # 6. Bouarfa 2024 anthropization - cultivated/fallow
    # --------------------------------------------------------------
    bou_raw = read_vector(
        cfg.bouarfa_2024_path,
        required=True,
    )

    bou_std = standardize_training_source(
        bou_raw,
        work_crs=work_crs,
        grid=field_grid,
        prefix="BOU",
        source="BOUARFA_ANTHRO_2024",
        class_name="Cultivated field/fallow",
        ref_year=2024,
        field_sup=0,
        confidence=3,
        geom_rule="EXISTING_CULTIVATED_POLYGON",
    )

    sources.append(final_schema(bou_std))
    source_counts["BOUARFA_ANTHRO_2024"] = len(bou_std)

    # --------------------------------------------------------------
    # 7. Selected HOT OSM built-up clusters
    # --------------------------------------------------------------
    built_raw = read_vector(
        cfg.builtup_path,
        layer=cfg.builtup_layer,
        required=True,
    )

    built_std = standardize_training_source(
        built_raw,
        work_crs=work_crs,
        grid=field_grid,
        prefix="BLD",
        source="HOT_OSM_BUILDINGS",
        class_name="Built-up",
        source_id_field="cluster_id",
        site_id_field="cluster_id",
        ref_year=2025,
        field_sup=0,
        confidence=2,
        geom_rule="OSM_BUILDING_CLUSTER",
    )

    sources.append(final_schema(built_std))
    source_counts["HOT_OSM_BUILDINGS"] = len(built_std)

    # --------------------------------------------------------------
    # 8. Al Baten areal candidates
    # --------------------------------------------------------------
    ab_areal_raw = read_vector(
        cfg.albaten_areal_path,
        layer=cfg.albaten_areal_layer,
        required=True,
    )

    ab_areal_std = standardize_training_source(
        ab_areal_raw,
        work_crs=work_crs,
        grid=field_grid,
        prefix="ABA",
        source="ALBATEN_INTERPRETED_MAP",
        class_field="class",
        source_id_field="part_id",
        site_id_field="part_id",
        field_sup=0,
        confidence=3,
        geom_rule="INTERPRETED_MAP_40M_PATCH",
    )

    sources.append(final_schema(ab_areal_std))
    source_counts["ALBATEN_INTERPRETED_MAP"] = len(ab_areal_std)

    # --------------------------------------------------------------
    # 9. Al Baten Wadi candidates
    # --------------------------------------------------------------
    ab_wadi_raw = read_vector(
        cfg.albaten_wadi_path,
        layer=cfg.albaten_wadi_layer,
        required=True,
    )

    ab_wadi_std = standardize_training_source(
        ab_wadi_raw,
        work_crs=work_crs,
        grid=field_grid,
        prefix="ABW",
        source="ALBATEN_INTERPRETED_MAP_WADI",
        class_name="Wadi and gullies",
        source_id_field="wadi_part_id",
        site_id_field="sample_id",
        field_sup=0,
        confidence=2,
        geom_rule="INTERPRETED_WADI_LOCAL_SEGMENT",
    )

    sources.append(final_schema(ab_wadi_std))
    source_counts["ALBATEN_INTERPRETED_MAP_WADI"] = len(ab_wadi_std)

    # --------------------------------------------------------------
    # 10. Merge
    # --------------------------------------------------------------
    training_candidates = gpd.GeoDataFrame(
        pd.concat(
            sources,
            ignore_index=True,
        ),
        geometry="geometry",
        crs=work_crs,
    )

    # --------------------------------------------------------------
    # 11. Sanity checks
    # --------------------------------------------------------------
    if training_candidates["sample_id"].duplicated().any():
        duplicated = training_candidates.loc[
            training_candidates["sample_id"].duplicated(),
            "sample_id",
        ].tolist()

        raise ValueError(
            f"Duplicate global sample IDs detected: {duplicated[:10]}"
        )

    if training_candidates["class"].isna().any():
        raise ValueError("Missing class labels detected after merge.")

    if training_candidates["class_id"].isna().any():
        raise ValueError("Missing class IDs detected after merge.")

    invalid_n = int((~training_candidates.geometry.is_valid).sum())
    empty_n = int(training_candidates.geometry.is_empty.sum())

    if invalid_n or empty_n:
        raise ValueError(
            f"Final merge contains invalid={invalid_n}, empty={empty_n} geometries."
        )

    # --------------------------------------------------------------
    # 12. Conflict flagging
    # --------------------------------------------------------------
    conflicts, conflict_ids = find_cross_class_conflicts(
        training_candidates,
        min_overlap_m2=cfg.min_conflict_overlap_m2,
    )

    training_candidates["conflict_flag"] = 0
    training_candidates.loc[
        training_candidates["sample_id"].isin(conflict_ids),
        "conflict_flag",
    ] = 1

    # --------------------------------------------------------------
    # 13. Pilot set
    # --------------------------------------------------------------
    training_pilot = make_pilot_set(
        training_candidates,
        conflict_policy=conflict_policy,
    )

    # --------------------------------------------------------------
    # 14. Summaries
    # --------------------------------------------------------------
    source_summary, class_summary = summarize_training(
        training_pilot
    )

    # --------------------------------------------------------------
    # 15. HOT OSM waterways - ancillary only
    # --------------------------------------------------------------
    waterways = read_vector(
        cfg.waterways_path,
        required=False,
    )

    if waterways is not None:
        waterways = waterways.to_crs(work_crs)

    # --------------------------------------------------------------
    # 16. Console report
    # --------------------------------------------------------------
    print("\nSource counts:")
    for source_name, n in source_counts.items():
        print(f"  {source_name:<32} {n:>6}")

    print(f"\nTotal training candidates: {len(training_candidates)}")
    print(f"Pilot training polygons:   {len(training_pilot)}")
    print(f"Cross-class conflict pairs:{len(conflicts):>7}")
    print(f"Conflict polygons:         {len(conflict_ids):>7}")
    print(f"Conflict policy:           {conflict_policy}")

    print("\nPilot class counts:")
    print(
        training_pilot["class"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------------
    # 17. Export
    # --------------------------------------------------------------
    if export:
        export_pipeline_outputs(
            output_path=cfg.output_path,
            training_candidates=training_candidates,
            training_pilot=training_pilot,
            field_grid=field_grid,
            source_summary=source_summary,
            class_summary=class_summary,
            conflicts=conflicts,
            conflict_ids=conflict_ids,
            waterways=waterways,
        )

        print(f"\nExported master database:\n{cfg.output_path}")

    # --------------------------------------------------------------
    # 18. Maps
    # --------------------------------------------------------------
    if make_plots:
        plot_training_maps(
            training_pilot,
            field_grid,
        )

    return {
        "training_candidates": training_candidates,
        "training_pilot": training_pilot,
        "source_summary": source_summary,
        "class_summary": class_summary,
        "conflicts": conflicts,
        "grid": field_grid,
        "waterways": waterways,
        "source_counts": source_counts,
        "config": cfg,
    }


# =====================================================================
# OPTIONAL COMMAND-LINE ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    run_pipeline(
        PipelineConfig(),
        conflict_policy="flag_only",
        export=True,
        make_plots=False,
    )
