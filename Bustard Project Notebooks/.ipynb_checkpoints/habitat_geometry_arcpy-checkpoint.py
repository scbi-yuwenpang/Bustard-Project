"""
Generate habitat polygons from the Bustard Project field Excel table.

Designed for ArcGIS Pro / ArcPy.

Geometry rules
--------------
1. Area_M2 like "5 x 250" -> centered rectangle, width=5 m, length=250 m.
   If an optional Azimuth_deg field is absent, the rectangle is temporarily
   oriented north-south and flagged for manual rotation.
2. Numeric Area_M2 like 750 -> centered equivalent-area square.
3. Blank Area_M2 -> 30 x 30 m Sentinel-2 sampling window.
   If SNAP_RASTER is provided, the window is aligned to the raster's 10 m grid.
4. Original GPS points are preserved in a separate feature class.
5. Features requiring inspection are copied to a review-points feature class.

Important assumptions
---------------------
- Latitude/Longitude are WGS 84 decimal degrees.
- The GPS point represents the CENTER of the field plot.
- Exposure is slope aspect and is NOT used as plot orientation.
- Add an Azimuth_deg column (clockwise from north) for true oriented rectangles.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import arcpy
import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================

INPUT_TABLE = r"C:\YourFolder\Phytosociological_habitat_2026.xlsx"
SHEET_NAME = 0  # sheet name or zero-based sheet index

OUTPUT_GDB = r"C:\YourFolder\Bustard_Habitat.gdb"
OUTPUT_PREFIX = "Habitat_2026"

# Eastern Morocco points in the supplied table fall within UTM Zone 30N.
OUTPUT_EPSG = 32630  # WGS 84 / UTM zone 30N

DEFAULT_WINDOW_M = 30.0
MIN_S2_WIDTH_M = 20.0

# Optional 10 m Sentinel-2 raster. Leave as None to create a centered 30 x 30 m
# square. When supplied, the raster must use the same spatial reference as
# OUTPUT_EPSG.
SNAP_RASTER = None
# Example:
# SNAP_RASTER = r"C:\YourFolder\Sentinel2_10m_reference.tif"

# Optional field containing plot/transect orientation, clockwise from north.
# The current table does not contain this field.
AZIMUTH_FIELD = "Azimuth_deg"

# The uploaded data appear to use the GPS point as a representative station
# location. Confirm this assumption before using reconstructed rectangles.
GPS_ROLE_DEFAULT = "CENTER"

OVERWRITE_OUTPUTS = True
ADD_OUTPUTS_TO_CURRENT_MAP = True


# =============================================================================
# STANDARDIZED HABITAT CLASSES
# =============================================================================

CLASS_ID_MAP = {
    "Grass steppe": 1,
    "Shrub steppe": 2,
    "Wooded steppe": 3,
    "Bare rocky terrain": 4,
    "Spreading area": 5,
    "Salty steppe": 6,
    "Dune": 7,
    "Wadi and gullies": 8,
    "Cultivated and fallow land": 9,
    "Daya and water body": 10,
    "Built-up area": 11,
}

# Default 30 m windows for these classes should always be inspected or replaced
# by imagery-based delineation when possible.
BOUNDARY_SENSITIVE_CLASSES = {
    "Wadi and gullies",
    "Daya and water body",
    "Cultivated and fallow land",
    "Built-up area",
}


# =============================================================================
# HELPERS
# =============================================================================

def log(message: str) -> None:
    print(message)
    try:
        arcpy.AddMessage(message)
    except Exception:
        pass


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def to_float(value: Any) -> Optional[float]:
    text = clean_text(value).replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_area(value: Any) -> Dict[str, Optional[float]]:
    """
    Parse Area_M2.

    Returns:
        kind: "dimensions", "area", or "missing"
        width_m, length_m, area_m2
    """
    text = clean_text(value).lower().replace("×", "x").replace("*", "x")

    if not text:
        return {
            "kind": "missing",
            "width_m": None,
            "length_m": None,
            "area_m2": None,
        }

    # Handles "5 x 250", "10x400", and optional meter labels.
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*(?:m)?\s*x\s*"
        r"(\d+(?:\.\d+)?)\s*(?:m)?\s*",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        d1 = float(match.group(1))
        d2 = float(match.group(2))
        width_m = min(d1, d2)
        length_m = max(d1, d2)
        return {
            "kind": "dimensions",
            "width_m": width_m,
            "length_m": length_m,
            "area_m2": width_m * length_m,
        }

    numeric_area = to_float(text)
    if numeric_area is not None and numeric_area > 0:
        side = math.sqrt(numeric_area)
        return {
            "kind": "area",
            "width_m": side,
            "length_m": side,
            "area_m2": numeric_area,
        }

    raise ValueError(f"Unrecognized Area_M2 value: {value!r}")


def infer_observation_type(station_id: Any) -> str:
    text = clean_text(station_id).lower()

    if text.startswith("phytosociology relev"):
        return "FIELD_RELEVE"
    if "visual interpretation" in text:
        return "IMAGE_INTERP"
    if "field observation" in text:
        return "FIELD_OBS"
    return "OTHER"


def standardize_class(row: pd.Series) -> Tuple[str, str]:
    """
    Return (standardized_class, correction_note).

    The supplied table contains Dune records whose class field is
    "Daya and water body" while Habitat_Full is "Dune". Habitat_Full overrides
    the class field for this known inconsistency.
    """
    original = clean_text(row.get("class"))
    habitat_full = clean_text(row.get("Habitat_Full"))
    habitat_lower = habitat_full.lower()

    if habitat_lower == "dune" or habitat_lower.startswith("dune>"):
        if original != "Dune":
            return "Dune", f"Class corrected from '{original}' using Habitat_Full='Dune'."
        return "Dune", ""

    if original in CLASS_ID_MAP:
        return original, ""

    # Conservative fallback rules for records with a missing/variant class.
    if "rocky outcrop" in habitat_lower or "scree" in habitat_lower:
        return "Bare rocky terrain", "Class inferred from Habitat_Full."
    if "gull" in habitat_lower or habitat_lower == "wadi":
        return "Wadi and gullies", "Class inferred from Habitat_Full."
    if habitat_lower == "fallow" or "cultivat" in habitat_lower:
        return "Cultivated and fallow land", "Class inferred from Habitat_Full."
    if habitat_lower == "daya":
        return "Daya and water body", "Class inferred from Habitat_Full."

    return original if original else "Unclassified", "Class could not be standardized."


def parse_azimuth(row: pd.Series) -> Optional[float]:
    """
    Read true plot/transect azimuth if the optional field exists.

    Exposure is deliberately ignored because it is slope aspect, not necessarily
    the orientation of the sampled rectangle.
    """
    if AZIMUTH_FIELD not in row.index:
        return None

    value = to_float(row.get(AZIMUTH_FIELD))
    if value is None:
        return None

    return value % 360.0


def make_rectangle(
    center_x: float,
    center_y: float,
    length_m: float,
    width_m: float,
    azimuth_deg: float,
    spatial_reference: arcpy.SpatialReference,
) -> arcpy.Polygon:
    """
    Create a rectangle centered on a point.

    Azimuth convention: degrees clockwise from north.
    """
    theta = math.radians(azimuth_deg)

    # Unit vector along length: 0 degrees = north.
    ux = math.sin(theta)
    uy = math.cos(theta)

    # Perpendicular unit vector across width.
    vx = math.cos(theta)
    vy = -math.sin(theta)

    half_l = length_m / 2.0
    half_w = width_m / 2.0

    corners = [
        (
            center_x - half_l * ux - half_w * vx,
            center_y - half_l * uy - half_w * vy,
        ),
        (
            center_x + half_l * ux - half_w * vx,
            center_y + half_l * uy - half_w * vy,
        ),
        (
            center_x + half_l * ux + half_w * vx,
            center_y + half_l * uy + half_w * vy,
        ),
        (
            center_x - half_l * ux + half_w * vx,
            center_y - half_l * uy + half_w * vy,
        ),
    ]

    points = [arcpy.Point(x, y) for x, y in corners]
    points.append(points[0])
    return arcpy.Polygon(arcpy.Array(points), spatial_reference)


def prepare_snap_grid(
    raster_path: Optional[str],
    output_sr: arcpy.SpatialReference,
) -> Optional[Dict[str, float]]:
    if not raster_path:
        return None

    if not arcpy.Exists(raster_path):
        raise FileNotFoundError(f"SNAP_RASTER does not exist: {raster_path}")

    desc = arcpy.Describe(raster_path)
    raster_sr = desc.spatialReference

    if not raster_sr or raster_sr.name == "Unknown":
        raise ValueError("SNAP_RASTER has an unknown spatial reference.")

    if raster_sr.factoryCode != output_sr.factoryCode:
        raise ValueError(
            "SNAP_RASTER must use the same spatial reference as OUTPUT_EPSG. "
            f"Raster={raster_sr.name}; output={output_sr.name}"
        )

    cell_x = float(
        arcpy.management.GetRasterProperties(raster_path, "CELLSIZEX").getOutput(0)
    )
    cell_y = abs(
        float(arcpy.management.GetRasterProperties(raster_path, "CELLSIZEY").getOutput(0))
    )

    return {
        "xmin": desc.extent.XMin,
        "ymin": desc.extent.YMin,
        "xmax": desc.extent.XMax,
        "ymax": desc.extent.YMax,
        "cell_x": cell_x,
        "cell_y": cell_y,
    }


def make_three_by_three_window(
    center_x: float,
    center_y: float,
    spatial_reference: arcpy.SpatialReference,
    snap_grid: Optional[Dict[str, float]],
) -> Tuple[arcpy.Polygon, float, float, str]:
    """
    Create either:
      - a 3 x 3 raster-cell window aligned to SNAP_RASTER, or
      - a centered DEFAULT_WINDOW_M square.
    """
    if snap_grid is None:
        geometry = make_rectangle(
            center_x,
            center_y,
            DEFAULT_WINDOW_M,
            DEFAULT_WINDOW_M,
            0.0,
            spatial_reference,
        )
        return (
            geometry,
            DEFAULT_WINDOW_M,
            DEFAULT_WINDOW_M,
            "DEFAULT_30M_CENTERED",
        )

    xmin = snap_grid["xmin"]
    ymin = snap_grid["ymin"]
    xmax = snap_grid["xmax"]
    ymax = snap_grid["ymax"]
    cell_x = snap_grid["cell_x"]
    cell_y = snap_grid["cell_y"]

    if not (xmin <= center_x <= xmax and ymin <= center_y <= ymax):
        raise ValueError("Point falls outside SNAP_RASTER extent.")

    col = math.floor((center_x - xmin) / cell_x)
    row = math.floor((center_y - ymin) / cell_y)

    containing_cell_center_x = xmin + (col + 0.5) * cell_x
    containing_cell_center_y = ymin + (row + 0.5) * cell_y

    length_m = 3.0 * cell_y
    width_m = 3.0 * cell_x

    geometry = make_rectangle(
        containing_cell_center_x,
        containing_cell_center_y,
        length_m,
        width_m,
        0.0,
        spatial_reference,
    )

    return geometry, length_m, width_m, "S2_3X3_ALIGNED"


def create_output_gdb(gdb_path: str) -> None:
    if arcpy.Exists(gdb_path):
        return

    folder = os.path.dirname(gdb_path)
    gdb_name = os.path.basename(gdb_path)

    if not folder:
        raise ValueError("OUTPUT_GDB must include a parent folder.")
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Output folder does not exist: {folder}")

    arcpy.management.CreateFileGDB(folder, gdb_name)


FIELD_DEFINITIONS = [
    ("SRC_ROW", "LONG", None),
    ("PLOT_ID", "TEXT", 50),
    ("STATION_N", "TEXT", 50),
    ("OBS_TYPE", "TEXT", 20),
    ("OBS_DATE", "TEXT", 30),
    ("LAT_DD", "DOUBLE", None),
    ("LON_DD", "DOUBLE", None),
    ("CLASS_ID", "SHORT", None),
    ("CLASS_NAME", "TEXT", 60),
    ("ORIG_CLASS", "TEXT", 60),
    ("HAB_FULL", "TEXT", 255),
    ("AREA_RAW", "TEXT", 40),
    ("AREA_M2", "DOUBLE", None),
    ("LENGTH_M", "DOUBLE", None),
    ("WIDTH_M", "DOUBLE", None),
    ("AZIMUTH", "DOUBLE", None),
    ("GPS_ROLE", "TEXT", 15),
    ("GEOM_TYPE", "TEXT", 30),
    ("GEOM_SRC", "TEXT", 30),
    ("GEOM_CONF", "TEXT", 12),
    ("MANUAL_QC", "SHORT", None),
    ("ROTATE_REQ", "SHORT", None),
    ("S2_USE", "TEXT", 20),
    ("ACT_AREA", "DOUBLE", None),
    ("QC_NOTE", "TEXT", 500),
]


def add_output_fields(feature_class: str) -> None:
    existing = {field.name.upper() for field in arcpy.ListFields(feature_class)}

    for name, field_type, length in FIELD_DEFINITIONS:
        if name.upper() in existing:
            continue

        kwargs = {}
        if field_type == "TEXT" and length is not None:
            kwargs["field_length"] = length

        arcpy.management.AddField(feature_class, name, field_type, **kwargs)


def create_feature_class(
    gdb: str,
    name: str,
    geometry_type: str,
    spatial_reference: arcpy.SpatialReference,
) -> str:
    output = os.path.join(gdb, name)

    if arcpy.Exists(output):
        if OVERWRITE_OUTPUTS:
            arcpy.management.Delete(output)
        else:
            raise FileExistsError(f"Output already exists: {output}")

    arcpy.management.CreateFeatureclass(
        out_path=gdb,
        out_name=name,
        geometry_type=geometry_type,
        spatial_reference=spatial_reference,
    )
    add_output_fields(output)
    return output


def read_input_table(path: str, sheet_name: Any) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        data = pd.read_excel(path, sheet_name=sheet_name, dtype=object)
    elif suffix == ".csv":
        data = pd.read_csv(path, dtype=object)
    elif suffix in {".txt", ".tsv"}:
        data = pd.read_csv(path, sep="\t", dtype=object)
    else:
        raise ValueError(
            "Unsupported input format. Use .xlsx, .xls, .xlsm, .csv, .txt, or .tsv."
        )

    data.columns = [clean_text(column) for column in data.columns]

    required = {"Latitude", "Longitude", "Area_M2", "class", "Habitat_Full"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    return data


def build_attributes(
    row: pd.Series,
    source_row: int,
    lat: float,
    lon: float,
    class_name: str,
    area_raw: str,
    area_m2: float,
    length_m: float,
    width_m: float,
    azimuth: Optional[float],
    geometry_type: str,
    geometry_source: str,
    geometry_confidence: str,
    manual_qc: int,
    rotate_required: int,
    s2_use: str,
    actual_area: float,
    qc_note: str,
) -> list:
    station_name = clean_text(row.get("Station_Name"))
    plot_id = station_name if station_name else f"OBS_{source_row:04d}"

    return [
        source_row,
        plot_id,
        station_name,
        infer_observation_type(row.get("Station_ID")),
        clean_text(row.get("Station_Date")),
        lat,
        lon,
        CLASS_ID_MAP.get(class_name),
        class_name,
        clean_text(row.get("class")),
        clean_text(row.get("Habitat_Full")),
        area_raw,
        area_m2,
        length_m,
        width_m,
        azimuth,
        GPS_ROLE_DEFAULT,
        geometry_type,
        geometry_source,
        geometry_confidence,
        manual_qc,
        rotate_required,
        s2_use,
        actual_area,
        qc_note[:500],
    ]


def main() -> None:
    arcpy.env.overwriteOutput = OVERWRITE_OUTPUTS

    if not os.path.isfile(INPUT_TABLE):
        raise FileNotFoundError(f"INPUT_TABLE does not exist: {INPUT_TABLE}")

    output_sr = arcpy.SpatialReference(OUTPUT_EPSG)
    wgs84 = arcpy.SpatialReference(4326)
    snap_grid = prepare_snap_grid(SNAP_RASTER, output_sr)

    create_output_gdb(OUTPUT_GDB)

    polygon_fc = create_feature_class(
        OUTPUT_GDB,
        f"{OUTPUT_PREFIX}_polygons",
        "POLYGON",
        output_sr,
    )
    source_point_fc = create_feature_class(
        OUTPUT_GDB,
        f"{OUTPUT_PREFIX}_source_points",
        "POINT",
        output_sr,
    )
    review_point_fc = create_feature_class(
        OUTPUT_GDB,
        f"{OUTPUT_PREFIX}_review_points",
        "POINT",
        output_sr,
    )

    data = read_input_table(INPUT_TABLE, SHEET_NAME)

    attribute_fields = [definition[0] for definition in FIELD_DEFINITIONS]
    cursor_fields = ["SHAPE@"] + attribute_fields

    counters = Counter()
    skipped_rows = []

    with (
        arcpy.da.InsertCursor(polygon_fc, cursor_fields) as polygon_cursor,
        arcpy.da.InsertCursor(source_point_fc, cursor_fields) as source_cursor,
        arcpy.da.InsertCursor(review_point_fc, cursor_fields) as review_cursor,
    ):
        for dataframe_index, row in data.iterrows():
            # +2 gives the approximate Excel row number: one header row + 1-based rows.
            source_row = int(dataframe_index) + 2

            lat = to_float(row.get("Latitude"))
            lon = to_float(row.get("Longitude"))

            # This also ignores the appended Center/Border table if it remains
            # beneath the main table on the same worksheet.
            if lat is None or lon is None:
                skipped_rows.append((source_row, "Missing/invalid Latitude or Longitude"))
                counters["skipped_invalid_coordinates"] += 1
                continue

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                skipped_rows.append((source_row, "Coordinates outside valid ranges"))
                counters["skipped_invalid_coordinates"] += 1
                continue

            source_point_wgs84 = arcpy.PointGeometry(
                arcpy.Point(lon, lat),
                wgs84,
            )
            source_point = source_point_wgs84.projectAs(output_sr)
            center = source_point.firstPoint

            class_name, class_note = standardize_class(row)
            area_raw = clean_text(row.get("Area_M2"))
            observation_type = infer_observation_type(row.get("Station_ID"))

            manual_qc = 0
            rotate_required = 0
            notes = []

            if class_note:
                notes.append(class_note)
                manual_qc = 1

            try:
                area_info = parse_area(row.get("Area_M2"))
            except ValueError as exc:
                area_info = {
                    "kind": "missing",
                    "width_m": None,
                    "length_m": None,
                    "area_m2": None,
                }
                notes.append(str(exc))
                manual_qc = 1

            azimuth = parse_azimuth(row)

            if area_info["kind"] == "dimensions":
                width_m = float(area_info["width_m"])
                length_m = float(area_info["length_m"])
                area_m2 = float(area_info["area_m2"])

                if azimuth is None:
                    # Placeholder only. The output review layer makes these
                    # easy to select and rotate after an azimuth is supplied.
                    azimuth_used = 0.0
                    rotate_required = 1
                    manual_qc = 1
                    geometry_type = "RECTANGLE_PLACEHOLDER"
                    geometry_confidence = "LOW"
                    notes.append(
                        "Length x width recorded, but no Azimuth_deg is available; "
                        "rectangle is temporarily north-south and must be rotated."
                    )
                else:
                    azimuth_used = azimuth
                    geometry_type = "FIELD_RECTANGLE"
                    geometry_confidence = "HIGH"

                geometry = make_rectangle(
                    center.X,
                    center.Y,
                    length_m,
                    width_m,
                    azimuth_used,
                    output_sr,
                )
                geometry_source = "FIELD_DIMENSIONS"

                if width_m < MIN_S2_WIDTH_M:
                    s2_use = "CONDITIONAL"
                    manual_qc = 1
                    notes.append(
                        f"Plot width ({width_m:g} m) is below the "
                        f"{MIN_S2_WIDTH_M:g} m Sentinel-2 purity threshold."
                    )
                else:
                    s2_use = "INCLUDE"

                azimuth_output = azimuth_used
                counters["field_rectangles"] += 1

            elif area_info["kind"] == "area":
                area_m2 = float(area_info["area_m2"])
                side_m = math.sqrt(area_m2)
                length_m = side_m
                width_m = side_m
                azimuth_output = 0.0

                geometry = make_rectangle(
                    center.X,
                    center.Y,
                    side_m,
                    side_m,
                    0.0,
                    output_sr,
                )
                geometry_type = "AREA_SQUARE"
                geometry_source = "FIELD_AREA"
                geometry_confidence = "MEDIUM"
                s2_use = "INCLUDE"
                notes.append(
                    "Square side inferred as sqrt(Area_M2); the original plot "
                    "shape was not recorded."
                )
                counters["area_squares"] += 1

                if class_name in BOUNDARY_SENSITIVE_CLASSES:
                    manual_qc = 1
                    s2_use = "CONDITIONAL"
                    notes.append(
                        "Boundary-sensitive class: inspect the inferred square "
                        "against high-resolution imagery."
                    )

            else:
                geometry, length_m, width_m, geometry_source = (
                    make_three_by_three_window(
                        center.X,
                        center.Y,
                        output_sr,
                        snap_grid,
                    )
                )
                area_m2 = length_m * width_m
                azimuth_output = 0.0
                geometry_type = "S2_SAMPLE_WINDOW"
                geometry_confidence = "LOW"
                manual_qc = 1
                counters["default_windows"] += 1

                notes.append(
                    "No field area was recorded; geometry is a remote-sensing "
                    "sampling window, not a measured field-plot footprint."
                )

                if class_name in BOUNDARY_SENSITIVE_CLASSES:
                    s2_use = "MANUAL_DIGITIZE"
                    notes.append(
                        "Replace with an imagery-delineated feature where possible."
                    )
                else:
                    s2_use = "CONDITIONAL"

            if observation_type == "IMAGE_INTERP":
                manual_qc = 1
                if s2_use == "INCLUDE":
                    s2_use = "CONDITIONAL"
                notes.append(
                    "Point originates from satellite-image interpretation rather "
                    "than an on-site field observation."
                )

            if class_name == "Unclassified":
                manual_qc = 1
                s2_use = "EXCLUDE"
                notes.append("Standardized class is missing.")

            actual_area = geometry.getArea("PLANAR", "SQUAREMETERS")
            qc_note = " ".join(notes)

            attributes = build_attributes(
                row=row,
                source_row=source_row,
                lat=lat,
                lon=lon,
                class_name=class_name,
                area_raw=area_raw,
                area_m2=area_m2,
                length_m=length_m,
                width_m=width_m,
                azimuth=azimuth_output,
                geometry_type=geometry_type,
                geometry_source=geometry_source,
                geometry_confidence=geometry_confidence,
                manual_qc=manual_qc,
                rotate_required=rotate_required,
                s2_use=s2_use,
                actual_area=actual_area,
                qc_note=qc_note,
            )

            polygon_cursor.insertRow([geometry] + attributes)
            source_cursor.insertRow([source_point] + attributes)

            if manual_qc:
                review_cursor.insertRow([source_point] + attributes)
                counters["manual_qc"] += 1

            counters["created"] += 1
            counters[f"class_{class_name}"] += 1

    log("Geometry generation completed.")
    log(f"Created polygons: {counters['created']}")
    log(f"  Field dimension rectangles: {counters['field_rectangles']}")
    log(f"  Equivalent-area squares: {counters['area_squares']}")
    log(f"  Default/S2 windows: {counters['default_windows']}")
    log(f"Features requiring manual QC: {counters['manual_qc']}")
    log(f"Rows skipped for invalid coordinates: {counters['skipped_invalid_coordinates']}")
    log(f"Polygon feature class: {polygon_fc}")
    log(f"Source point feature class: {source_point_fc}")
    log(f"Review point feature class: {review_point_fc}")

    if skipped_rows:
        log("First skipped rows:")
        for row_number, reason in skipped_rows[:10]:
            log(f"  Row {row_number}: {reason}")

    if ADD_OUTPUTS_TO_CURRENT_MAP:
        try:
            project = arcpy.mp.ArcGISProject("CURRENT")
            active_map = project.activeMap
            if active_map is not None:
                active_map.addDataFromPath(polygon_fc)
                active_map.addDataFromPath(review_point_fc)
                active_map.addDataFromPath(source_point_fc)
                log("Outputs added to the active ArcGIS Pro map.")
        except Exception as exc:
            log(f"Outputs were created but not added to the current map: {exc}")


if __name__ == "__main__":
    main()