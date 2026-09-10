#!/usr/bin/env python
"""
preprocess_albaten_areal_habitats.py

Reusable preprocessing workflow for areal habitat classes in the
Al Baten expert-interpreted habitat map.

Workflow
--------
1. Read original habitat polygons.
2. Harmonize habitat labels to the final mapping legend.
3. Separate Wadi and Built-up from areal habitat classes.
4. Explode multipart areal features into individual habitat patches.
5. Remove an inward edge zone to reduce boundary/mixed-pixel effects.
6. Remove tiny safe interiors.
7. Sample a limited number of spatially separated square training polygons
   from homogeneous interiors.
8. Attach provenance / class / optional regional grid metadata.
9. Export review-ready GeoPackage layers and CSV summaries.

The script deliberately does NOT use all pixels from large polygons.
Large interpreted polygons are capped to a few spatially separated
training patches to reduce pseudoreplication.

Example
-------
python preprocess_albaten_areal_habitats.py
  --input "C:\\path\\al_baten_habitat_mapping_polygon.shp"
  --output "C:\\path\\Al_Baten_Training_2025.gpkg"
  --grid "C:\\path\\Eastern_Morocco_Training_Grid.gpkg"
  --grid-layer grid_20km
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box


# ---------------------------------------------------------------------
# EDIT THIS CROSSWALK IF THE ORIGINAL MAP CONTAINS ADDITIONAL LABELS.
# The script stops when it finds an unmapped non-null label so that
# classes are never silently discarded.
# ---------------------------------------------------------------------
CLASS_MAP = {
    "Alfah-grass steppe":"Grass steppe",
    "Perennial herbaceous cultivation":"Grass steppe",
    "Alluvial silt plain":"Spreading area",
    "Reg":"Bare rocky",
    "Salty steppe":"Salty steppe",
    "Tree farming":"Cultivated field/fallow",
    "Wadi":"Wadi and gullies",
    "Flooded wadi":"Wadi and gullies",
    "Building":"Built-up"
}

CLASS_ID = {
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess Al Baten areal habitat polygons into review-ready training patches."
    )
    p.add_argument("--input", required=True, help="Original Al Baten habitat polygon file.")
    p.add_argument("--output", required=True, help="Output GeoPackage path.")
    p.add_argument("--class-field", default="Habitat", help="Original habitat-label field.")
    p.add_argument("--grid", default=None, help="Optional regional grid file/GPKG.")
    p.add_argument("--grid-layer", default=None, help="Optional layer name for a grid GeoPackage.")
    p.add_argument("--grid-id-field", default="grid_id", help="Grid identifier field.")
    p.add_argument("--work-crs", default=None,
                   help="Optional projected CRS, e.g. EPSG:32630. If omitted, grid CRS or estimated UTM is used.")

    p.add_argument("--edge-buffer", type=float, default=15.0,
                   help="Inward edge exclusion distance in meters.")
    p.add_argument("--patch-size", type=float, default=40.0,
                   help="Square training-patch side length in meters.")
    p.add_argument("--min-safe-area", type=float, default=2500.0,
                   help="Minimum safe interior area in m2.")
    p.add_argument("--min-separation", type=float, default=300.0,
                   help="Minimum distance between patch centers from the same source part in meters.")
    p.add_argument("--max-patches-per-part", type=int, default=5,
                   help="Maximum sampled patches from a single interpreted habitat part.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    return p.parse_args()


def choose_work_crs(gdf, grid=None, requested=None):
    if requested:
        return requested
    if grid is not None and grid.crs is not None:
        return grid.crs
    estimated = gdf.estimate_utm_crs()
    if estimated is None:
        raise ValueError(
            "Could not estimate a projected CRS. Supply --work-crs explicitly."
        )
    return estimated


def valid_nonempty_geometry(g):
    return g is not None and (not g.is_empty) and g.is_valid


def assign_grid(gdf, grid, grid_id_field):
    if grid is None:
        out = gdf.copy()
        out["grid_id"] = pd.NA
        return out

    if grid_id_field not in grid.columns:
        raise KeyError(f"Grid field '{grid_id_field}' not found.")

    temp = gdf.reset_index(drop=True).copy()

    pts = gpd.GeoDataFrame(
        {"_rowid": np.arange(len(temp))},
        geometry=temp.geometry.representative_point(),
        crs=temp.crs,
    )

    joined = gpd.sjoin(
        pts,
        grid[[grid_id_field, "geometry"]],
        how="left",
        predicate="within",
    )

    joined = (
        joined.sort_values("_rowid")
        .drop_duplicates("_rowid", keep="first")
        .set_index("_rowid")
    )

    temp["grid_id"] = joined[grid_id_field].reindex(
        np.arange(len(temp))
    ).values

    return temp


def samples_for_area(area_m2, max_patches):
    if area_m2 < 50_000:
        return 1
    if area_m2 < 250_000:
        return min(2, max_patches)
    if area_m2 < 1_000_000:
        return min(3, max_patches)
    return max_patches


def sample_square_patches(
    polygon,
    n,
    patch_size,
    min_distance,
    max_attempts,
    seed,
):
    """
    Generate square polygons fully contained in a source polygon.

    min_distance is measured between square centers.
    """
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = polygon.bounds
    half = patch_size / 2.0

    selected_centers = []
    selected_squares = []
    attempts = 0

    while len(selected_squares) < n and attempts < max_attempts:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        center = Point(x, y)
        attempts += 1

        if not polygon.contains(center):
            continue

        square = box(
            x - half,
            y - half,
            x + half,
            y + half,
        )

        # The entire training patch must remain inside the safe interior.
        if not polygon.covers(square):
            continue

        if selected_centers:
            too_close = any(
                center.distance(existing) < min_distance
                for existing in selected_centers
            )
            if too_close:
                continue

        selected_centers.append(center)
        selected_squares.append(square)

    return selected_squares


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Start with a clean GeoPackage on each rerun.
    if output_path.exists():
        output_path.unlink()

    raw = gpd.read_file(input_path)

    if args.class_field not in raw.columns:
        raise KeyError(
            f"Class field '{args.class_field}' not found. "
            f"Available fields: {raw.columns.tolist()}"
        )

    grid = None
    if args.grid:
        if args.grid_layer:
            grid = gpd.read_file(args.grid, layer=args.grid_layer)
        else:
            grid = gpd.read_file(args.grid)

    work_crs = choose_work_crs(raw, grid=grid, requested=args.work_crs)
    raw = raw.to_crs(work_crs).reset_index(drop=True)

    if grid is not None:
        grid = grid.to_crs(work_crs)

    raw["orig_id"] = [
        f"AB_ORIG_{i:03d}" for i in range(1, len(raw) + 1)
    ]

    raw["class"] = raw[args.class_field].map(CLASS_MAP)

    non_null_labels = set(raw[args.class_field].dropna().astype(str))
    mapped_labels = set(CLASS_MAP.keys())
    unmapped = sorted(non_null_labels - mapped_labels)

    if unmapped:
        raise ValueError(
            "Unmapped habitat labels detected. Update CLASS_MAP before continuing:\n  - "
            + "\n  - ".join(unmapped)
        )

    # Save special classes separately for later dedicated workflows.
    builtup = raw[raw["class"] == "Built-up"].copy()
    wadi = raw[raw["class"] == "Wadi and gullies"].copy()

    areal = raw[
        raw["class"].notna()
        & ~raw["class"].isin(["Built-up", "Wadi and gullies"])
    ].copy()

    parts = (
        areal.explode(index_parts=False)
        .reset_index(drop=True)
    )

    parts["part_id"] = [
        f"AB_PART_{i:05d}" for i in range(1, len(parts) + 1)
    ]
    parts["part_area_m2"] = parts.geometry.area

    # Inward edge removal.
    safe = parts.copy()
    n_before = len(safe)

    safe["geometry"] = safe.geometry.buffer(-args.edge_buffer)

    geom_ok = safe.geometry.apply(valid_nonempty_geometry)
    safe = safe[geom_ok].copy()
    safe["safe_area_m2"] = safe.geometry.area

    n_after_edge = len(safe)

    # Class-level retention table before minimum-area filter.
    before_class = parts["class"].value_counts().rename("n_before")
    after_class = safe["class"].value_counts().rename("n_after_edge")
    edge_summary = pd.concat(
        [before_class, after_class], axis=1
    ).fillna(0)

    edge_summary["n_removed_edge"] = (
        edge_summary["n_before"] - edge_summary["n_after_edge"]
    )
    edge_summary["retained_pct"] = np.where(
        edge_summary["n_before"] > 0,
        edge_summary["n_after_edge"] / edge_summary["n_before"] * 100.0,
        np.nan,
    )

    usable = safe[
        safe["safe_area_m2"] >= args.min_safe_area
    ].copy()

    training_records = []

    for row_number, (_, row) in enumerate(usable.iterrows()):
        n_samples = samples_for_area(
            row["safe_area_m2"],
            args.max_patches_per_part,
        )

        squares = sample_square_patches(
            polygon=row.geometry,
            n=n_samples,
            patch_size=args.patch_size,
            min_distance=args.min_separation,
            max_attempts=10_000,
            seed=args.seed + row_number,
        )

        for patch_no, geom in enumerate(squares, start=1):
            training_records.append(
                {
                    "orig_id": row["orig_id"],
                    "part_id": row["part_id"],
                    "class_raw": row[args.class_field],
                    "class": row["class"],
                    "source_part_area_m2": row["part_area_m2"],
                    "safe_area_m2": row["safe_area_m2"],
                    "patch_no": patch_no,
                    "geometry": geom,
                }
            )

    training = gpd.GeoDataFrame(
        training_records,
        geometry="geometry",
        crs=work_crs,
    )

    if len(training) == 0:
        raise RuntimeError(
            "No training patches were generated. "
            "Check the class mapping and sampling thresholds."
        )

    training = training.reset_index(drop=True)
    training["sample_id"] = [
        f"ABTR_{i:05d}" for i in range(1, len(training) + 1)
    ]
    training["site_id"] = training["part_id"]
    training["source_id"] = training["part_id"]
    training["class_id"] = training["class"].map(CLASS_ID)
    training["source"] = "ALBATEN_INTERPRETED_MAP"
    training["ref_year"] = pd.NA
    training["field_sup"] = 0
    training["confidence"] = 3
    training["geom_rule"] = (
        f"INTERPRETED_MAP_{int(args.patch_size)}M_PATCH"
    )
    training["qc_status"] = "UNREVIEWED"
    training["use_status"] = "CANDIDATE"
    training["poly_area"] = training.geometry.area

    training = assign_grid(
        training,
        grid=grid,
        grid_id_field=args.grid_id_field,
    )

    sampled_parts = set(training["part_id"])
    usable["sample_generated"] = usable["part_id"].isin(sampled_parts)

    sampling_success = (
        usable.groupby("class")["sample_generated"]
        .agg(n_parts="size", n_sampled="sum")
    )
    sampling_success["success_pct"] = np.where(
        sampling_success["n_parts"] > 0,
        sampling_success["n_sampled"] / sampling_success["n_parts"] * 100.0,
        np.nan,
    )

    training_summary = (
        training.groupby("class")
        .agg(
            n_polygons=("sample_id", "size"),
            n_sites=("site_id", "nunique"),
            n_grids=("grid_id", "nunique"),
            total_area_m2=("poly_area", "sum"),
            median_area_m2=("poly_area", "median"),
        )
        .sort_values("n_polygons", ascending=False)
    )

    # Export review-ready layers.
    raw.to_file(output_path, layer="source_original", driver="GPKG")
    parts.to_file(output_path, layer="areal_parts", driver="GPKG")
    safe.to_file(output_path, layer="safe_interiors", driver="GPKG")
    usable.to_file(output_path, layer="usable_source_patches", driver="GPKG")
    training.to_file(output_path, layer="training_patches", driver="GPKG")

    if len(wadi):
        wadi.to_file(output_path, layer="wadi_review", driver="GPKG")
    if len(builtup):
        builtup.to_file(output_path, layer="builtup_review", driver="GPKG")

    # CSV diagnostics beside the GeoPackage.
    stem = output_path.with_suffix("")
    edge_summary.to_csv(
        str(stem) + "_edge_retention.csv"
    )
    sampling_success.to_csv(
        str(stem) + "_sampling_success.csv"
    )
    training_summary.to_csv(
        str(stem) + "_training_summary.csv"
    )

    print("\n=== Al Baten areal habitat preprocessing complete ===")
    print(f"Working CRS: {work_crs}")
    print(f"Areal parts before edge removal: {n_before}")
    print(f"Safe parts after edge removal:  {n_after_edge}")
    print(
        "Removed by edge erosion:        "
        f"{n_before - n_after_edge}"
    )
    print(f"Usable safe source patches:     {len(usable)}")
    print(f"Generated training polygons:    {len(training)}")
    print(f"\nOutput: {output_path}")
    print("\nTraining polygons by class:")
    print(training["class"].value_counts())


if __name__ == "__main__":
    main()
