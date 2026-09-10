#!/usr/bin/env python
r"""
merge_eastern_morocco_training_final.py

Final merge of all reviewed/preprocessed Eastern Morocco habitat-training
sources for the 2025 pilot classification.

This script DOES NOT redo source-specific preprocessing. It expects the
prepared outputs already created for:
  - phytosociological field footprints
  - Hayat field-point 10 m pilot polygons
  - Trophic field-point 10 m pilot polygons
  - reviewed Al Baten + Bouarfa 2024 cultivated polygons
  - selected HOT OSM building clusters
  - Al Baten areal training patches
  - Al Baten local Wadi segments
  - Tamlelt interpreted habitat candidates

HOT OSM waterway lines are retained as an ancillary reference layer only
and are NOT merged into the polygon training set.

Typical Jupyter use
-------------------
import sys
sys.path.append(r"C:\Users\PangY\OneDrive - Smithsonian Institution\Bustard\07_Bustard-Project\Bustard Project Notebooks")

import merge_eastern_morocco_training_final as mt

result = mt.build_final_training(
    conflict_policy="flag_only",
    export=True,
    make_plots=True
)

training_final = result["training_final"]
class_summary = result["class_summary"]
source_summary = result["source_summary"]
conflicts = result["conflicts"]

display(class_summary)
display(source_summary)

Conflict policies
-----------------
"flag_only"
    Keep all reviewed/prepared samples for the first pilot, but add
    conflict_flag=1 to polygons involved in cross-class overlaps.

"exclude_all"
    Remove all polygons involved in cross-class overlaps from training_final.
    The full flagged dataset is still exported for traceability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    **{k: k for k in CLASS_DICT},
    "Open woodland": "Wooded steppe",
    "Sand": "Dune",
    "Cutlivated field/fallow": "Cultivated field/fallow",
    "Wadi & gullies": "Wadi and gullies",
    "Wadi and gully": "Wadi and gullies",
    "Reg": "Bare rocky",
    "Alfah-grass steppe": "Grass steppe",
    "Alfa-grass steppe": "Grass steppe",
    "Chamaephyte steppe": "Shrub steppe",
    "Anabasis steppe": "Shrub steppe",
    "Alluvial silt plain": "Spreading area",
    "Tree farming": "Cultivated field/fallow",
    "Wadi": "Wadi and gullies",
    "Flooded wadi": "Wadi and gullies",
    "Building": "Built-up",
}

FINAL_FIELDS = [
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
    "source_qc_status",
    "source_use_status",
    "qc_status",
    "use_status",
    "poly_area",
    "conflict_flag",
    "geometry",
]


@dataclass
class FinalMergeConfig:
    base: Path = Path(
        r"C:\Users\PangY\OneDrive - Smithsonian Institution\Bustard\01_Data"
    )

    phyto_path: Optional[Path] = None
    phyto_layer: str = "field_footprints"

    field_supplement_path: Optional[Path] = None
    hayat_layer: str = "hayat_training_std"
    trophic_layer: str = "trophic_training_std"

    cultivated_path: Optional[Path] = None
    albaten_farms_layer: str = "albaten_farms_2024"
    bouarfa_2024_layer: str = "bouarfa_cultivated_2024"

    builtup_path: Optional[Path] = None
    builtup_layer: str = "builtup_selected"

    albaten_areal_path: Optional[Path] = None
    albaten_areal_layer: str = "training_patches"

    albaten_wadi_path: Optional[Path] = None
    albaten_wadi_layer: str = "draft_wadi_segments"

    tamlelt_path: Optional[Path] = None
    tamlelt_layer: str = "training_candidates"

    grid_path: Optional[Path] = None
    grid_layer: str = "grid_20km"

    waterways_path: Optional[Path] = None

    output_path: Optional[Path] = None

    min_conflict_overlap_m2: float = 25.0

    def __post_init__(self):
        train_dir = (
            self.base
            / r"Reneco\Vegetation Cover and Food Availability Studies"
        )
        habitat_dir = self.base / "Habitat mapping"
        anthro_dir = (
            self.base
            / r"Reneco\GIS Data\Extract data Yuwen\Shapefiles\anthropisation"
        )

        if self.phyto_path is None:
            self.phyto_path = (
                train_dir / "Eastern_Morocco_Field_Footprints.gpkg"
            )

        if self.field_supplement_path is None:
            self.field_supplement_path = (
                habitat_dir / "Field_Supplement_Training_2025.gpkg"
            )

        if self.cultivated_path is None:
            self.cultivated_path = (
                anthro_dir / "Eastern_Morocco_Cultivated_Training_2024.gpkg"
            )

        if self.builtup_path is None:
            self.builtup_path = (
                habitat_dir
                / "build-up"
                / "Eastern_Morocco_BuiltUp_Candidates_2025.gpkg"
            )

        if self.albaten_areal_path is None:
            self.albaten_areal_path = (
                habitat_dir / "Al_Baten_Training_2025.gpkg"
            )

        if self.albaten_wadi_path is None:
            self.albaten_wadi_path = (
                habitat_dir / "Al_Baten_Wadi_Training_2025.gpkg"
            )

        if self.tamlelt_path is None:
            self.tamlelt_path = (
                habitat_dir / "Tamlelt_Training_2025.gpkg"
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
                train_dir / "Eastern_Morocco_Training_Final_2025.gpkg"
            )


def read_source(
    path: Path,
    layer: Optional[str] = None,
    required: bool = True,
) -> Optional[gpd.GeoDataFrame]:
    path = Path(path)

    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing source: {path}")
        return None

    if path.suffix.lower() == ".gpkg" and layer:
        return gpd.read_file(path, layer=layer)

    return gpd.read_file(path)


def clean_label(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace(CLASS_CROSSWALK)
    )


def clean_polygon_geometry(
    gdf: gpd.GeoDataFrame,
    target_crs,
) -> gpd.GeoDataFrame:
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
    out = out[keep].copy()

    geom_types = set(out.geometry.geom_type.unique())
    allowed = {"Polygon", "MultiPolygon"}

    if not geom_types.issubset(allowed):
        raise ValueError(
            f"Expected polygon training geometry, found: {sorted(geom_types)}"
        )

    return out


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


def first_existing_field(
    gdf: gpd.GeoDataFrame,
    candidates: List[str],
) -> Optional[str]:
    for field in candidates:
        if field in gdf.columns:
            return field
    return None


def drop_explicit_exclusions(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if "use_status" not in gdf.columns:
        return gdf.copy()

    status = (
        gdf["use_status"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    exclude = status.str.startswith("EXCLUDE", na=False)

    if exclude.any():
        print(f"    honoring source exclusions: {int(exclude.sum())}")

    return gdf[~exclude].copy()


def standardize_source(
    gdf: gpd.GeoDataFrame,
    *,
    work_crs,
    field_grid: gpd.GeoDataFrame,
    prefix: str,
    source_name: str,
    class_field: Optional[str] = "class",
    fixed_class: Optional[str] = None,
    source_id_candidates: Optional[List[str]] = None,
    site_id_candidates: Optional[List[str]] = None,
    ref_year: Optional[int] = None,
    field_sup: int = 0,
    confidence: int = 2,
    default_geom_rule: str = "SOURCE_POLYGON",
) -> gpd.GeoDataFrame:

    out = drop_explicit_exclusions(gdf)
    out = clean_polygon_geometry(out, work_crs)
    out = out.reset_index(drop=True)

    if "qc_status" in out.columns:
        out["source_qc_status"] = out["qc_status"].astype("string")
    else:
        out["source_qc_status"] = pd.NA

    if "use_status" in out.columns:
        out["source_use_status"] = out["use_status"].astype("string")
    else:
        out["source_use_status"] = pd.NA

    if fixed_class is not None:
        # Preserve source-side raw class/provenance if it already exists.
        if "class_raw" not in out.columns:
            out["class_raw"] = fixed_class
        out["class"] = fixed_class
    else:
        if class_field is None or class_field not in out.columns:
            raise KeyError(
                f"{source_name}: required class field '{class_field}' not found. "
                f"Available fields: {out.columns.tolist()}"
            )

        if "class_raw" not in out.columns:
            out["class_raw"] = (
                out[class_field]
                .astype("string")
                .str.strip()
            )

        out["class"] = clean_label(out[class_field])

    unmapped = sorted(
        set(
            out.loc[
                ~out["class"].isin(CLASS_DICT),
                "class",
            ]
            .dropna()
            .astype(str)
        )
    )

    if unmapped:
        raise ValueError(
            f"{source_name}: labels not mapped to final legend:\n  - "
            + "\n  - ".join(unmapped)
        )

    if out["class"].isna().any():
        raise ValueError(
            f"{source_name}: null class values remain after harmonization."
        )

    out["class_id"] = out["class"].map(CLASS_DICT)

    source_id_candidates = source_id_candidates or [
        "source_id",
        "part_id",
        "wadi_part_id",
        "cluster_id",
        "Station_ID",
        "Station_na",
        "Station",
        "sample_id",
        "orig_id",
        "OBJECTID",
        "FID",
    ]

    sid_field = first_existing_field(out, source_id_candidates)

    if sid_field is not None:
        source_ids = out[sid_field].astype("string")
    else:
        source_ids = pd.Series(
            out.index.astype(str),
            index=out.index,
            dtype="string",
        )

    source_ids = source_ids.fillna(
        pd.Series(
            out.index.astype(str),
            index=out.index,
            dtype="string",
        )
    )

    out["source_id"] = source_ids

    site_id_candidates = site_id_candidates or [
        "site_id",
        "Station_ID",
        "Station_na",
        "Station",
        "part_id",
        "wadi_part_id",
        "cluster_id",
        "source_id",
    ]

    site_field = first_existing_field(out, site_id_candidates)

    if site_field is not None:
        site_ids = out[site_field].astype("string")
    else:
        site_ids = out["source_id"].astype("string")

    site_ids = site_ids.fillna(out["source_id"].astype("string"))

    out["sample_id"] = [
        f"{prefix}_{i:05d}"
        for i in range(1, len(out) + 1)
    ]

    out["site_id"] = site_ids.apply(
        lambda x: x if str(x).startswith(prefix + "_") else prefix + "_" + str(x)
    )

    out["source"] = source_name

    # Preserve an existing source-specific reference year.
    # Only fill missing values when a default ref_year was actually supplied.
    if "ref_year" not in out.columns:
        out["ref_year"] = ref_year if ref_year is not None else pd.NA
    elif ref_year is not None:
        out["ref_year"] = out["ref_year"].fillna(ref_year)

    if "field_sup" not in out.columns:
        out["field_sup"] = field_sup
    else:
        out["field_sup"] = out["field_sup"].fillna(field_sup)

    if "confidence" not in out.columns:
        out["confidence"] = confidence
    else:
        out["confidence"] = out["confidence"].fillna(confidence)

    if "geom_rule" not in out.columns:
        out["geom_rule"] = default_geom_rule
    else:
        out["geom_rule"] = out["geom_rule"].fillna(default_geom_rule)

    out["qc_status"] = "FINAL_READY"
    out["use_status"] = "TRAIN"

    out["poly_area"] = out.geometry.area
    out["conflict_flag"] = 0

    out = assign_grid(out, field_grid)

    return out


def final_schema(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()

    for col in FINAL_FIELDS:
        if col not in out.columns:
            out[col] = pd.NA

    return out[FINAL_FIELDS].copy()


def find_cross_class_conflicts(
    training: gpd.GeoDataFrame,
    min_overlap_m2: float = 25.0,
) -> Tuple[pd.DataFrame, set]:

    sindex = training.sindex
    records: List[Dict[str, Any]] = []
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

            intersection = row_i.geometry.intersection(
                row_j.geometry
            )

            if intersection.is_empty:
                continue

            overlap_area = intersection.area

            if overlap_area < min_overlap_m2:
                continue

            area_i = row_i.geometry.area
            area_j = row_j.geometry.area

            records.append(
                {
                    "sample_1": row_i["sample_id"],
                    "class_1": row_i["class"],
                    "source_1": row_i["source"],
                    "sample_2": row_j["sample_id"],
                    "class_2": row_j["class"],
                    "source_2": row_j["source"],
                    "overlap_m2": overlap_area,
                    "overlap_pct_1": (
                        overlap_area / area_i * 100.0
                        if area_i > 0 else np.nan
                    ),
                    "overlap_pct_2": (
                        overlap_area / area_j * 100.0
                        if area_j > 0 else np.nan
                    ),
                }
            )

            conflict_ids.add(row_i["sample_id"])
            conflict_ids.add(row_j["sample_id"])

    conflicts = pd.DataFrame(records)

    if len(conflicts):
        conflicts = (
            conflicts
            .sort_values("overlap_m2", ascending=False)
            .reset_index(drop=True)
        )

    return conflicts, conflict_ids


def summarize_training(
    training: gpd.GeoDataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    source_summary = (
        training
        .groupby(["source", "class"])
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
        training
        .groupby("class")
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


def plot_final_training(
    training: gpd.GeoDataFrame,
    field_grid: gpd.GeoDataFrame,
):
    fig, ax = plt.subplots(figsize=(15, 11))

    field_grid.boundary.plot(
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
        "Eastern Morocco - Final Training Polygons",
        fontsize=15,
    )
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    sites = training.copy()
    sites["geometry"] = sites.geometry.representative_point()

    fig, ax = plt.subplots(figsize=(15, 11))

    field_grid.boundary.plot(
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
        "Eastern Morocco - Spatial Distribution of Final Training Sites",
        fontsize=15,
    )
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def build_final_training(
    cfg: Optional[FinalMergeConfig] = None,
    *,
    conflict_policy: str = "flag_only",
    export: bool = True,
    make_plots: bool = True,
) -> Dict[str, Any]:

    if cfg is None:
        cfg = FinalMergeConfig()

    if conflict_policy not in {"flag_only", "exclude_all"}:
        raise ValueError(
            "conflict_policy must be 'flag_only' or 'exclude_all'."
        )

    print("=" * 76)
    print("EASTERN MOROCCO - FINAL TRAINING DATA MERGE")
    print("=" * 76)

    field_grid = read_source(
        cfg.grid_path,
        layer=cfg.grid_layer,
        required=True,
    )

    if field_grid.crs is None:
        raise ValueError("Regional grid has no CRS.")

    work_crs = field_grid.crs

    print(f"Working CRS: {work_crs}")
    print(f"20-km grid cells: {len(field_grid)}")

    source_layers: List[gpd.GeoDataFrame] = []
    source_counts: Dict[str, int] = {}

    print("\n[1/9] Phytosociological footprints")
    raw = read_source(cfg.phyto_path, layer=cfg.phyto_layer)
    std = standardize_source(
        raw,
        work_crs=work_crs,
        field_grid=field_grid,
        prefix="PHY",
        source_name="PHYTOSOC",
        class_field="class",
        source_id_candidates=["Station_ID", "source_id", "sample_id"],
        site_id_candidates=["Station_ID", "site_id"],
        field_sup=1,
        confidence=2,
        default_geom_rule="AREA_EQUIV_BUFFER",
    )
    source_layers.append(final_schema(std))
    source_counts["PHYTOSOC"] = len(std)
    print(f"    retained: {len(std)}")

    print("\n[2/9] Hayat field supplement")
    raw = read_source(
        cfg.field_supplement_path,
        layer=cfg.hayat_layer,
    )
    std = standardize_source(
        raw,
        work_crs=work_crs,
        field_grid=field_grid,
        prefix="HAY",
        source_name="HAYAT_PHYTO",
        class_field="class",
        source_id_candidates=["source_id", "Station_na", "sample_id"],
        site_id_candidates=["site_id", "Station_na"],
        field_sup=1,
        confidence=2,
        default_geom_rule="POINT_BUFFER_10M_PILOT",
    )
    source_layers.append(final_schema(std))
    source_counts["HAYAT_PHYTO"] = len(std)
    print(f"    retained: {len(std)}")

    print("\n[3/9] Trophic field supplement")
    raw = read_source(
        cfg.field_supplement_path,
        layer=cfg.trophic_layer,
    )
    std = standardize_source(
        raw,
        work_crs=work_crs,
        field_grid=field_grid,
        prefix="TRO",
        source_name="TROPHIC_SURVEY",
        class_field="class",
        source_id_candidates=["source_id", "Station", "sample_id"],
        site_id_candidates=["site_id", "Station"],
        field_sup=1,
        confidence=2,
        default_geom_rule="POINT_BUFFER_10M_PILOT",
    )
    source_layers.append(final_schema(std))
    source_counts["TROPHIC_SURVEY"] = len(std)
    print(f"    retained: {len(std)}")

    print("\n[4/9] Reviewed Al Baten cultivated polygons")
    raw = read_source(
        cfg.cultivated_path,
        layer=cfg.albaten_farms_layer,
    )
    std = standardize_source(
        raw,
        work_crs=work_crs,
        field_grid=field_grid,
        prefix="ABF",
        source_name="ALBATEN_FARMS_2024",
        fixed_class="Cultivated field/fallow",
        ref_year=2024,
        field_sup=0,
        confidence=3,
        default_geom_rule="EXISTING_CULTIVATED_POLYGON",
    )
    source_layers.append(final_schema(std))
    source_counts["ALBATEN_FARMS_2024"] = len(std)
    print(f"    retained: {len(std)}")

    print("\n[5/9] Reviewed Bouarfa 2024 cultivated polygons")
    raw = read_source(
        cfg.cultivated_path,
        layer=cfg.bouarfa_2024_layer,
    )
    std = standardize_source(
        raw,
        work_crs=work_crs,
        field_grid=field_grid,
        prefix="BOU",
        source_name="BOUARFA_ANTHRO_2024",
        fixed_class="Cultivated field/fallow",
        ref_year=2024,
        field_sup=0,
        confidence=3,
        default_geom_rule="EXISTING_CULTIVATED_POLYGON",
    )
    source_layers.append(final_schema(std))
    source_counts["BOUARFA_ANTHRO_2024"] = len(std)
    print(f"    retained: {len(std)}")

    print("\n[6/9] HOT OSM built-up candidates")
    raw = read_source(cfg.builtup_path, layer=cfg.builtup_layer)
    std = standardize_source(
        raw,
        work_crs=work_crs,
        field_grid=field_grid,
        prefix="BLD",
        source_name="HOT_OSM_BUILDINGS",
        fixed_class="Built-up",
        source_id_candidates=["cluster_id", "source_id", "sample_id"],
        site_id_candidates=["cluster_id", "site_id"],
        ref_year=2025,
        field_sup=0,
        confidence=2,
        default_geom_rule="OSM_BUILDING_CLUSTER",
    )
    source_layers.append(final_schema(std))
    source_counts["HOT_OSM_BUILDINGS"] = len(std)
    print(f"    retained: {len(std)}")

    print("\n[7/9] Al Baten areal habitat candidates")
    raw = read_source(
        cfg.albaten_areal_path,
        layer=cfg.albaten_areal_layer,
    )
    std = standardize_source(
        raw,
        work_crs=work_crs,
        field_grid=field_grid,
        prefix="ABA",
        source_name="ALBATEN_INTERPRETED_MAP",
        class_field="class",
        source_id_candidates=["part_id", "source_id", "sample_id"],
        site_id_candidates=["part_id", "site_id"],
        field_sup=0,
        confidence=3,
        default_geom_rule="INTERPRETED_MAP_40M_PATCH",
    )
    source_layers.append(final_schema(std))
    source_counts["ALBATEN_INTERPRETED_MAP"] = len(std)
    print(f"    retained: {len(std)}")

    print("\n[8/9] Al Baten Wadi candidates")
    raw = read_source(
        cfg.albaten_wadi_path,
        layer=cfg.albaten_wadi_layer,
    )
    std = standardize_source(
        raw,
        work_crs=work_crs,
        field_grid=field_grid,
        prefix="ABW",
        source_name="ALBATEN_INTERPRETED_MAP_WADI",
        fixed_class="Wadi and gullies",
        source_id_candidates=["wadi_part_id", "source_id", "sample_id"],
        site_id_candidates=["site_id", "sample_id", "wadi_part_id"],
        field_sup=0,
        confidence=2,
        default_geom_rule="INTERPRETED_WADI_LOCAL_SEGMENT",
    )
    source_layers.append(final_schema(std))
    source_counts["ALBATEN_INTERPRETED_MAP_WADI"] = len(std)
    print(f"    retained: {len(std)}")

    print("\n[9/9] Tamlelt interpreted habitat candidates")
    raw = read_source(cfg.tamlelt_path, layer=cfg.tamlelt_layer)
    std = standardize_source(
        raw,
        work_crs=work_crs,
        field_grid=field_grid,
        prefix="TAM",
        source_name="TAMLELT_INTERPRETED_MAP",
        class_field="class",
        source_id_candidates=["part_id", "source_id", "sample_id"],
        site_id_candidates=["site_id", "part_id"],
        field_sup=0,
        confidence=3,
        default_geom_rule="INTERPRETED_POLYGON",
    )
    source_layers.append(final_schema(std))
    source_counts["TAMLELT_INTERPRETED_MAP"] = len(std)
    print(f"    retained: {len(std)}")

    training_all = gpd.GeoDataFrame(
        pd.concat(
            source_layers,
            ignore_index=True,
        ),
        geometry="geometry",
        crs=work_crs,
    )

    training_all["poly_area"] = training_all.geometry.area

    if training_all["sample_id"].duplicated().any():
        raise ValueError("Duplicate global sample_id values detected.")

    if training_all["class"].isna().any():
        raise ValueError("Null class labels detected in final merge.")

    if training_all["class_id"].isna().any():
        raise ValueError("Null class_id values detected in final merge.")

    invalid_n = int((~training_all.geometry.is_valid).sum())
    empty_n = int(training_all.geometry.is_empty.sum())

    if invalid_n or empty_n:
        raise ValueError(
            f"Final merge has invalid={invalid_n}, empty={empty_n} geometries."
        )

    conflicts, conflict_ids = find_cross_class_conflicts(
        training_all,
        min_overlap_m2=cfg.min_conflict_overlap_m2,
    )

    training_all["conflict_flag"] = 0
    training_all.loc[
        training_all["sample_id"].isin(conflict_ids),
        "conflict_flag",
    ] = 1

    if conflict_policy == "flag_only":
        training_final = training_all.copy()
    else:
        training_final = training_all[
            training_all["conflict_flag"] == 0
        ].copy()

    training_final = training_final.reset_index(drop=True)

    source_summary, class_summary = summarize_training(
        training_final
    )

    print("\n" + "=" * 76)
    print("FINAL MERGE SUMMARY")
    print("=" * 76)

    print("\nSource counts:")
    for source_name, n in source_counts.items():
        print(f"  {source_name:<34} {n:>6}")

    print(f"\nAll prepared polygons:      {len(training_all)}")
    print(f"Final training polygons:    {len(training_final)}")
    print(f"Cross-class conflict pairs: {len(conflicts)}")
    print(f"Polygons with conflicts:    {len(conflict_ids)}")
    print(f"Conflict policy:            {conflict_policy}")

    print("\nFinal class counts:")
    print(
        training_final["class"]
        .value_counts()
        .to_string()
    )

    waterways = read_source(
        cfg.waterways_path,
        required=False,
    )

    if waterways is not None:
        waterways = waterways.to_crs(work_crs)

    if export:
        output_path = Path(cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            try:
                output_path.unlink()
            except PermissionError as exc:
                raise PermissionError(
                    f"Cannot overwrite {output_path}. "
                    "Close the GeoPackage in ArcGIS Pro and rerun."
                ) from exc

        training_all.to_file(
            output_path,
            layer="training_all_flagged",
            driver="GPKG",
        )

        training_final.to_file(
            output_path,
            layer="training_final",
            driver="GPKG",
        )

        field_grid.to_file(
            output_path,
            layer="grid_20km",
            driver="GPKG",
        )

        if len(conflict_ids):
            conflict_features = training_all[
                training_all["sample_id"].isin(conflict_ids)
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

        base_no_suffix = output_path.with_suffix("")

        source_summary.to_csv(
            str(base_no_suffix) + "_source_summary.csv",
            index=False,
        )

        class_summary.to_csv(
            str(base_no_suffix) + "_class_summary.csv",
        )

        if len(conflicts):
            conflicts.to_csv(
                str(base_no_suffix) + "_conflicts.csv",
                index=False,
            )

        print(f"\nSaved:\n{output_path}")
        print("Main model-ready layer: training_final")

    if make_plots:
        plot_final_training(
            training_final,
            field_grid,
        )

    return {
        "training_final": training_final,
        "training_all_flagged": training_all,
        "class_summary": class_summary,
        "source_summary": source_summary,
        "conflicts": conflicts,
        "grid": field_grid,
        "waterways": waterways,
        "source_counts": source_counts,
        "config": cfg,
    }


if __name__ == "__main__":
    build_final_training(
        FinalMergeConfig(),
        conflict_policy="flag_only",
        export=True,
        make_plots=False,
    )
