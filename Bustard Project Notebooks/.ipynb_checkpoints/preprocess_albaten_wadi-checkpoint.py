#!/usr/bin/env python
"""
preprocess_albaten_wadi.py

Reusable preprocessing workflow for the linear Wadi and gullies class in
the Al Baten expert-interpreted habitat map.

Methodological logic
--------------------
- HOT OSM waterway lines are used ONLY to guide WHERE representative
  Wadi locations should be sampled.
- The original expert-interpreted Al Baten Wadi polygons remain the
  source of habitat geometry.
- Selected anchor locations are spatially thinned and stratified within
  a local grid.
- A local review window is created around each anchor.
- The original expert Wadi polygon is clipped to each local window to
  obtain a short draft Wadi segment.
- Draft segments are exported for VHR/ArcGIS review and edge refinement.

This avoids arbitrary wide line buffers and avoids using one enormous
merged Wadi polygon as a training unit.

Example
-------
python preprocess_albaten_wadi.py
  --habitat-input "C:\\path\\al_baten_habitat_mapping_polygon.shp"
  --water-lines "C:\\path\\hotosm_mar_waterways_lines_EM.shp"
  --output "C:\\path\\Al_Baten_Wadi_Training_2025.gpkg"
  --grid "C:\\path\\Eastern_Morocco_Training_Grid.gpkg"
  --grid-layer grid_20km
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


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
        description="Create review-ready local Wadi training segments for Al Baten."
    )
    p.add_argument("--habitat-input", required=True,
                   help="Original Al Baten habitat polygon map.")
    p.add_argument("--water-lines", required=True,
                   help="Canonical HOT OSM waterway line layer.")
    p.add_argument("--output", required=True,
                   help="Output GeoPackage path.")
    p.add_argument("--class-field", default="Habitat",
                   help="Habitat-label field in the original polygon map.")
    p.add_argument("--waterway-field", default="waterway",
                   help="Waterway-type field in HOT OSM line layer.")
    p.add_argument("--grid", default=None,
                   help="Optional regional 20-km grid.")
    p.add_argument("--grid-layer", default=None,
                   help="Optional layer name for a grid GeoPackage.")
    p.add_argument("--grid-id-field", default="grid_id")
    p.add_argument("--work-crs", default=None,
                   help="Optional projected CRS. If omitted, grid CRS or estimated UTM is used.")

    p.add_argument("--min-guide-length", type=float, default=50.0,
                   help="Minimum clipped OSM guide-line length in meters.")
    p.add_argument("--point-spacing", type=float, default=300.0,
                   help="Initial anchor spacing along guide lines in meters.")
    p.add_argument("--min-site-distance", type=float, default=300.0,
                   help="Minimum spacing among selected anchors in meters.")
    p.add_argument("--local-grid-size", type=float, default=2000.0,
                   help="Local sampling-grid cell size in meters.")
    p.add_argument("--max-per-local-grid", type=int, default=2,
                   help="Maximum selected Wadi anchors per local grid cell.")
    p.add_argument("--window-size", type=float, default=200.0,
                   help="Square review-window side length in meters.")
    p.add_argument("--min-wadi-area", type=float, default=400.0,
                   help="Minimum local draft Wadi polygon area in m2.")

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


def sample_points_along_line(line, spacing):
    length = line.length
    if length <= 0:
        return []

    if length < spacing:
        return [line.interpolate(0.5, normalized=True)]

    distances = np.arange(
        spacing / 2.0,
        length,
        spacing,
    )
    return [line.interpolate(d) for d in distances]


def thin_points(gdf, min_distance):
    """
    Greedy thinning, prioritizing candidate points originating from
    longer waterway guide segments.
    """
    if len(gdf) == 0:
        return gdf.copy()

    ordered = gdf.sort_values(
        "guide_length_m",
        ascending=False,
    ).copy()

    selected_idx = []
    selected_xy = []

    for idx, row in ordered.iterrows():
        x = row.geometry.x
        y = row.geometry.y

        if not selected_xy:
            selected_idx.append(idx)
            selected_xy.append((x, y))
            continue

        xy = np.asarray(selected_xy)
        distances = np.sqrt(
            (xy[:, 0] - x) ** 2
            + (xy[:, 1] - y) ** 2
        )

        if distances.min() >= min_distance:
            selected_idx.append(idx)
            selected_xy.append((x, y))

    return ordered.loc[selected_idx].copy()


def polygon_parts_from_geometry(geom):
    if geom is None or geom.is_empty:
        return []

    if geom.geom_type == "Polygon":
        return [geom]

    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)

    if geom.geom_type == "GeometryCollection":
        polygons = []
        for g in geom.geoms:
            if g.geom_type == "Polygon":
                polygons.append(g)
            elif g.geom_type == "MultiPolygon":
                polygons.extend(list(g.geoms))
        return polygons

    return []


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    habitat = gpd.read_file(args.habitat_input)
    water_lines = gpd.read_file(args.water_lines)

    if args.class_field not in habitat.columns:
        raise KeyError(
            f"Class field '{args.class_field}' not found in habitat map."
        )

    grid = None
    if args.grid:
        if args.grid_layer:
            grid = gpd.read_file(args.grid, layer=args.grid_layer)
        else:
            grid = gpd.read_file(args.grid)

    work_crs = choose_work_crs(
        habitat,
        grid=grid,
        requested=args.work_crs,
    )

    habitat = habitat.to_crs(work_crs).reset_index(drop=True)
    water_lines = water_lines.to_crs(work_crs).copy()

    if grid is not None:
        grid = grid.to_crs(work_crs)

    habitat["orig_id"] = [
        f"AB_ORIG_{i:03d}" for i in range(1, len(habitat) + 1)
    ]

    # Extract only interpreted Wadi classes.
    wadi_source = habitat[
        habitat[args.class_field].isin(["Wadi", "Flooded wadi"])
    ].copy()

    if len(wadi_source) == 0:
        raise RuntimeError(
            "No 'Wadi' or 'Flooded wadi' features were found."
        )

    wadi_source["class"] = "Wadi and gullies"

    wadi_parts = (
        wadi_source.explode(index_parts=False)
        .reset_index(drop=True)
    )

    wadi_parts["wadi_part_id"] = [
        f"AB_WADI_PART_{i:05d}"
        for i in range(1, len(wadi_parts) + 1)
    ]
    wadi_parts["part_area_m2"] = wadi_parts.geometry.area

    # Remove obvious artificial waterway types when the field exists.
    if args.waterway_field in water_lines.columns:
        water_lines["waterway_clean"] = (
            water_lines[args.waterway_field]
            .astype("string")
            .str.lower()
            .str.strip()
        )

        artificial = {"canal", "ditch", "drain"}

        guide_source = water_lines[
            ~water_lines["waterway_clean"].isin(artificial)
        ].copy()
    else:
        guide_source = water_lines.copy()

    # Clip HOT lines to the expert-interpreted Wadi footprint.
    # unary_union is intentionally used for compatibility with older
    # GeoPandas/Shapely installations.
    wadi_union = wadi_parts.geometry.unary_union

    wadi_mask = gpd.GeoDataFrame(
        {"mask_id": [1]},
        geometry=[wadi_union],
        crs=work_crs,
    )

    guides = gpd.overlay(
        guide_source,
        wadi_mask,
        how="intersection",
        keep_geom_type=False,
    )

    guides = (
        guides.explode(index_parts=False)
        .reset_index(drop=True)
    )

    guides = guides[
        guides.geometry.geom_type == "LineString"
    ].copy()

    guides["guide_id"] = [
        f"WG_{i:05d}" for i in range(1, len(guides) + 1)
    ]
    guides["guide_length_m"] = guides.geometry.length

    guides = guides[
        guides["guide_length_m"] >= args.min_guide_length
    ].copy()

    if len(guides) == 0:
        raise RuntimeError(
            "No usable HOT OSM guide lines intersect the interpreted Wadi polygons."
        )

    # Initial anchors along the guide network.
    candidate_records = []

    for _, row in guides.iterrows():
        pts = sample_points_along_line(
            row.geometry,
            spacing=args.point_spacing,
        )
        for p in pts:
            candidate_records.append(
                {
                    "guide_id": row["guide_id"],
                    "guide_length_m": row["guide_length_m"],
                    "geometry": p,
                }
            )

    candidates = gpd.GeoDataFrame(
        candidate_records,
        geometry="geometry",
        crs=work_crs,
    )

    thinned = thin_points(
        candidates,
        min_distance=args.min_site_distance,
    )

    # Local sampling grid for spatial coverage inside Al Baten.
    minx, miny, maxx, maxy = wadi_parts.total_bounds
    cells = []
    cell_no = 1

    for x in np.arange(minx, maxx, args.local_grid_size):
        for y in np.arange(miny, maxy, args.local_grid_size):
            cells.append(
                {
                    "local_grid": f"ABG_{cell_no:03d}",
                    "geometry": box(
                        x,
                        y,
                        x + args.local_grid_size,
                        y + args.local_grid_size,
                    ),
                }
            )
            cell_no += 1

    local_grid = gpd.GeoDataFrame(
        cells,
        geometry="geometry",
        crs=work_crs,
    )

    thinned = (
        gpd.sjoin(
            thinned,
            local_grid[["local_grid", "geometry"]],
            how="left",
            predicate="within",
        )
        .drop(columns="index_right", errors="ignore")
    )

    selected = (
        thinned.sort_values(
            ["local_grid", "guide_length_m"],
            ascending=[True, False],
        )
        .groupby("local_grid", group_keys=False)
        .head(args.max_per_local_grid)
        .copy()
        .reset_index(drop=True)
    )

    selected["sample_id"] = [
        f"ABW_{i:04d}" for i in range(1, len(selected) + 1)
    ]

    # Link each anchor back to an original interpreted Wadi component.
    anchor_source = gpd.sjoin(
        selected,
        wadi_parts[
            [
                "wadi_part_id",
                "orig_id",
                args.class_field,
                "geometry",
            ]
        ],
        how="left",
        predicate="intersects",
    ).drop(columns="index_right", errors="ignore")

    # In case an anchor touches multiple parts, retain one record.
    anchor_source = (
        anchor_source.sort_values("sample_id")
        .drop_duplicates("sample_id", keep="first")
        .reset_index(drop=True)
    )

    missing_links = anchor_source["wadi_part_id"].isna().sum()
    if missing_links:
        print(
            f"Warning: {missing_links} selected anchors did not link "
            "to an original Wadi polygon and will be skipped."
        )

    half_window = args.window_size / 2.0

    windows = selected.copy()
    windows["geometry"] = windows.geometry.apply(
        lambda p: box(
            p.x - half_window,
            p.y - half_window,
            p.x + half_window,
            p.y + half_window,
        )
    )
    windows = gpd.GeoDataFrame(
        windows,
        geometry="geometry",
        crs=work_crs,
    )

    part_lookup = wadi_parts.set_index("wadi_part_id")
    draft_records = []

    for _, row in anchor_source.iterrows():
        part_id = row["wadi_part_id"]

        if pd.isna(part_id):
            continue

        source_geom = part_lookup.loc[part_id].geometry
        point_geom = row.geometry

        window_geom = box(
            point_geom.x - half_window,
            point_geom.y - half_window,
            point_geom.x + half_window,
            point_geom.y + half_window,
        )

        clipped = source_geom.intersection(window_geom)
        pieces = polygon_parts_from_geometry(clipped)

        if not pieces:
            continue

        containing = [
            p for p in pieces
            if p.buffer(0.5).contains(point_geom)
        ]

        if containing:
            selected_geom = max(containing, key=lambda g: g.area)
        else:
            selected_geom = min(
                pieces,
                key=lambda g: g.distance(point_geom),
            )

        draft_records.append(
            {
                "sample_id": row["sample_id"],
                "wadi_part_id": part_id,
                "orig_id": row["orig_id"],
                "class_raw": row[args.class_field],
                "guide_id": row["guide_id"],
                "local_grid": row["local_grid"],
                "geometry": selected_geom,
            }
        )

    draft = gpd.GeoDataFrame(
        draft_records,
        geometry="geometry",
        crs=work_crs,
    )

    if len(draft) == 0:
        raise RuntimeError(
            "No local Wadi polygons were extracted."
        )

    draft["poly_area"] = draft.geometry.area

    draft = draft[
        draft["poly_area"] >= args.min_wadi_area
    ].copy().reset_index(drop=True)

    draft["class"] = "Wadi and gullies"
    draft["class_id"] = CLASS_ID["Wadi and gullies"]
    draft["site_id"] = draft["sample_id"]
    draft["source_id"] = draft["wadi_part_id"]
    draft["source"] = "ALBATEN_INTERPRETED_MAP"
    draft["ref_year"] = pd.NA
    draft["field_sup"] = 0
    draft["confidence"] = 2
    draft["geom_rule"] = "INTERPRETED_WADI_LOCAL_SEGMENT"
    draft["qc_status"] = "UNREVIEWED"
    draft["use_status"] = "CANDIDATE"

    draft = assign_grid(
        draft,
        grid=grid,
        grid_id_field=args.grid_id_field,
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "original_wadi_parts",
                "usable_osm_guide_segments",
                "initial_anchor_candidates",
                "anchors_after_spatial_thinning",
                "selected_anchor_locations",
                "draft_wadi_segments",
                "local_grid_cells_represented",
            ],
            "value": [
                len(wadi_parts),
                len(guides),
                len(candidates),
                len(thinned),
                len(selected),
                len(draft),
                selected["local_grid"].nunique(),
            ],
        }
    )

    # Export review-ready layers.
    wadi_parts.to_file(
        output_path,
        layer="original_wadi_parts",
        driver="GPKG",
    )
    guides.to_file(
        output_path,
        layer="osm_wadi_guides",
        driver="GPKG",
    )
    selected.to_file(
        output_path,
        layer="selected_locations",
        driver="GPKG",
    )
    local_grid.to_file(
        output_path,
        layer="local_sampling_grid",
        driver="GPKG",
    )
    windows.to_file(
        output_path,
        layer="review_windows",
        driver="GPKG",
    )
    draft.to_file(
        output_path,
        layer="draft_wadi_segments",
        driver="GPKG",
    )

    summary.to_csv(
        str(output_path.with_suffix("")) + "_summary.csv",
        index=False,
    )

    print("\n=== Al Baten Wadi preprocessing complete ===")
    print(f"Working CRS: {work_crs}")
    print(summary.to_string(index=False))
    print(f"\nOutput: {output_path}")
    print(
        "\nNext step: review 'draft_wadi_segments' in ArcGIS/VHR, "
        "reshape edges where necessary, and set qc_status/use_status."
    )


if __name__ == "__main__":
    main()
