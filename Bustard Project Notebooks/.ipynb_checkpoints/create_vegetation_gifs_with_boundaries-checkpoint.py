from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # Render without opening a GUI window.
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

# Folder containing the downloaded GEE GeoTIFFs.
INPUT_DIR = Path(r"C:\Users\PangY\OneDrive - Smithsonian Institution\Bustard\01_Data\GEE Satellite\GEE_Landsat_Annual_Export")

# Folder containing the shapefiles listed below. Keep each .shp together with
BOUNDARY_DIR = Path(r"C:\Users\PangY\OneDrive - Smithsonian Institution\Bustard\01_Data\Morocco two study areas")

# Output folder for the GIFs.
OUTPUT_DIR = INPUT_DIR / "GIFs"

START_YEAR = 2008
END_YEAR = 2025

# Animation speed in milliseconds per frame.
FRAME_DURATION_MS = 900
LAST_FRAME_DURATION_MS = 1800

# Reduce very large rasters only for animation display. The source TIFFs are
# not changed. Increase this for a sharper but larger/slower GIF.
MAX_RASTER_DIMENSION = 1600

# Output rendering resolution.
DPI = 140

# Select which products to create.
MAKE_INDIVIDUAL_GIFS = True
MAKE_SIDE_BY_SIDE_GIF = True

# This is the most similar to the ArcGIS screenshot: both prediction rasters
# are shown in their true geographic positions inside the Eastern Morocco
# boundary.
MAKE_EASTERN_MOROCCO_CONTEXT_GIF = True

# If True, stop when any year from START_YEAR to END_YEAR is missing.
# If False, create the GIF from the available years and print a warning.
REQUIRE_ALL_YEARS = False

# Used only if the TIFF does not contain NoData metadata.
NODATA_FALLBACK = -9999.0

REGIONS = {
    "Al_Baten": "Al Baten",
    "Bouarfa": "Bouarfa",
}

# Shapefiles. Edit only these names if your local filenames differ.
SHAPEFILES = {
    "Eastern_Morocco": "Eastern_Morocco_Working_Area.shp",
    "Al_Baten": "Al_Baten_study_area.shp",
    "Al_Baten_buffer": "Al_Baten_study_area_10kmBuffer.shp",
    "Bouarfa": "Bouarfa_study_area.shp",
    "Bouarfa_buffer": "Bouarfa_study_area_10kmBuffer.shp",
}

# -----------------------------------------------------------------------------
# Display scale and palette. Keep the same DISPLAY_MIN/MAX for every year so
# that changes in color represent real temporal changes rather than automatic
# per-frame stretching.
#
# If most predictions are below about 0.5 (as in your ArcGIS screenshot), you
# can set DISPLAY_MAX = 0.5 for stronger visual contrast, provided you are
# comfortable clipping values above 0.5 in the visualization only.
# -----------------------------------------------------------------------------
DISPLAY_MIN = 0.0
DISPLAY_MAX = 0.5 #instead of 1.0

#   "green"  -> pale green to dark green
PALETTE_STYLE = "green"

GREEN_PALETTE = [
    "#f7fcf5",
    "#e5f5e0",
    "#c7e9c0",
    "#a1d99b",
    "#74c476",
    "#41ab5d",
    "#238b45",
    "#006d2c",
    "#00441b",
]

ARCGIS_LIKE_PALETTE = [
    "#006837",
    "#31a354",
    "#78c679",
    "#c2e699",
    "#ffffb2",
    "#fecc5c",
    "#fd8d3c",
    "#f03b20",
    "#bd0026",
]

if PALETTE_STYLE.lower() == "arcgis":
    palette = ARCGIS_LIKE_PALETTE
    palette_name = "live_vegetation_arcgis_like"
elif PALETTE_STYLE.lower() == "green":
    palette = GREEN_PALETTE
    palette_name = "live_vegetation_green"
else:
    raise ValueError(
        f"Unknown PALETTE_STYLE: {PALETTE_STYLE}. Use green or arcgis."
    )

CMAP = LinearSegmentedColormap.from_list(palette_name, palette, N=256)
# Transparent NoData is especially useful in the Eastern Morocco context map.
CMAP.set_bad((1.0, 1.0, 1.0, 0.0))
NORM = Normalize(vmin=DISPLAY_MIN, vmax=DISPLAY_MAX, clip=True)

# Boundary styles. The Eastern Morocco line follows the blue-outline look in
# the ArcGIS screenshot; local study areas are solid black and 10-km buffers
# are dashed gray.
EASTERN_MOROCCO_COLOR = "#1f77b4"
STUDY_AREA_COLOR = "#111111"
BUFFER_COLOR = "#666666"

# Matches, for example:
# Landsat_RF_Al_Baten_2024_ANN_prediction.tif
FILE_PATTERN = re.compile(
    r"^Landsat_RF_(Al_Baten|Bouarfa)_(\d{4})_ANN_prediction\.(?:tif|tiff)$",
    flags=re.IGNORECASE,
)


# =============================================================================
# 2. INPUT DISCOVERY AND VALIDATION
# =============================================================================
def discover_rasters() -> dict[str, dict[int, Path]]:
    """Find prediction TIFFs and index them by region and year."""
    found: dict[str, dict[int, Path]] = {region: {} for region in REGIONS}
    canonical_region = {region.lower(): region for region in REGIONS}

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"INPUT_DIR does not exist: {INPUT_DIR}\n"
            "Edit INPUT_DIR near the top of this script."
        )

    for path in INPUT_DIR.rglob("*"):
        if not path.is_file():
            continue

        match = FILE_PATTERN.match(path.name)
        if match is None:
            continue

        region = canonical_region[match.group(1).lower()]
        year = int(match.group(2))

        if year in found[region]:
            raise RuntimeError(
                f"Duplicate raster for {region}, {year}:\n"
                f"  {found[region][year]}\n"
                f"  {path}"
            )

        found[region][year] = path

    return found


def selected_years(files_by_year: dict[int, Path], region_label: str) -> list[int]:
    """Return available years within the requested period and report gaps."""
    expected = set(range(START_YEAR, END_YEAR + 1))
    available = set(files_by_year).intersection(expected)
    missing = sorted(expected.difference(available))

    if missing:
        message = f"{region_label}: missing years: {missing}"
        if REQUIRE_ALL_YEARS:
            raise FileNotFoundError(message)
        print(f"WARNING - {message}")

    years = sorted(available)
    if not years:
        print(f"WARNING - No prediction rasters found for {region_label}.")

    return years


def validate_region_grid(
    files_by_year: dict[int, Path],
    years: list[int],
    region_label: str,
) -> None:
    """Ensure frames for one region share the same CRS, dimensions and bounds."""
    if not years:
        return

    reference = None
    reference_year = years[0]

    for year in years:
        with rasterio.open(files_by_year[year]) as src:
            signature = (
                src.crs,
                src.width,
                src.height,
                tuple(src.transform),
                tuple(src.bounds),
            )

        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(
                f"{region_label} raster grid differs between {reference_year} "
                f"and {year}. Re-export or align the rasters before animation."
            )


def get_raster_crs(path: Path):
    """Return the CRS from a raster and fail clearly if it is missing."""
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {path}")
        return src.crs


# =============================================================================
# 3. BOUNDARY READING
# =============================================================================
def load_boundaries() -> dict[str, gpd.GeoDataFrame]:
    """Read all requested shapefiles once."""
    if not BOUNDARY_DIR.exists():
        raise FileNotFoundError(
            f"BOUNDARY_DIR does not exist: {BOUNDARY_DIR}\n"
            "Edit BOUNDARY_DIR near the top of this script."
        )

    boundaries: dict[str, gpd.GeoDataFrame] = {}

    for key, filename in SHAPEFILES.items():
        path = BOUNDARY_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Boundary shapefile not found: {path}")

        gdf = gpd.read_file(path)
        if gdf.empty:
            raise ValueError(f"Boundary shapefile is empty: {path}")
        if gdf.crs is None:
            raise ValueError(
                f"Boundary shapefile has no CRS: {path}\n"
                "Make sure the .prj file is present."
            )

        # Remove empty geometries while preserving multipart features.
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        boundaries[key] = gdf

    return boundaries


def reproject_boundaries(
    boundaries: dict[str, gpd.GeoDataFrame],
    target_crs,
) -> dict[str, gpd.GeoDataFrame]:
    """Reproject all boundaries to the raster CRS."""
    return {
        key: gdf.to_crs(target_crs)
        for key, gdf in boundaries.items()
    }


def plot_region_boundaries(
    ax: plt.Axes,
    boundaries: dict[str, gpd.GeoDataFrame],
    region: str,
) -> None:
    """Overlay the focal study-area and 10-km-buffer boundaries."""
    boundaries[f"{region}_buffer"].boundary.plot(
        ax=ax,
        linewidth=1.4,
        linestyle="--",
        color=BUFFER_COLOR,
        zorder=5,
    )
    boundaries[region].boundary.plot(
        ax=ax,
        linewidth=2.0,
        linestyle="-",
        color=STUDY_AREA_COLOR,
        zorder=6,
    )


def add_region_label(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    label: str,
) -> None:
    """Place a simple label near the center of a study-area polygon."""
    geom = gdf.geometry.unary_union
    point = geom.representative_point()
    ax.text(
        point.x,
        point.y,
        label,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#111111",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
        },
        zorder=8,
    )


# =============================================================================
# 4. RASTER READING
# =============================================================================


def read_raster_for_display(
    path: Path,
) -> tuple[np.ma.MaskedArray, tuple[float, float, float, float]]:
    """Read one raster, mask NoData, and optionally downsample for display."""
    with rasterio.open(path) as src:
        scale = min(
            1.0,
            MAX_RASTER_DIMENSION / max(src.width, src.height),
        )
        out_width = max(1, int(round(src.width * scale)))
        out_height = max(1, int(round(src.height * scale)))

        # F_alive is continuous, so bilinear resampling is suitable for display.
        data = src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

        # Read the validity mask separately with nearest-neighbour resampling.
        invalid = src.read_masks(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.nearest,
        ) == 0

        nodata = src.nodata
        if nodata is None:
            nodata = NODATA_FALLBACK

        invalid |= ~np.isfinite(data)
        invalid |= np.isclose(data, nodata)
        invalid |= (data < 0.0) | (data > 1.0)
        data = np.clip(data, 0.0, 1.0)

        masked_data = np.ma.array(data, mask=invalid)
        extent = (
            src.bounds.left,
            src.bounds.right,
            src.bounds.bottom,
            src.bounds.top,
        )

    return masked_data, extent


# =============================================================================
# 5. FRAME RENDERING
# =============================================================================


def figure_to_pillow_image(fig: plt.Figure) -> Image.Image:
    """Convert a Matplotlib figure to an in-memory Pillow RGB image."""
    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=DPI,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)

    buffer.seek(0)
    with Image.open(buffer) as image:
        frame = image.convert("RGB").copy()

    buffer.close()
    return frame


def add_colorbar(fig: plt.Figure, cax: plt.Axes) -> None:
    """Add the common fixed 0-1 color scale."""
    scalar_mappable = ScalarMappable(norm=NORM, cmap=CMAP)
    scalar_mappable.set_array([])

    colorbar = fig.colorbar(scalar_mappable, cax=cax)
    colorbar.set_ticks(np.linspace(DISPLAY_MIN, DISPLAY_MAX, 6))
    colorbar.set_label(
        "Predicted fractional live vegetation cover (F_alive)",
        fontsize=10,
        labelpad=10,
    )
    colorbar.ax.tick_params(labelsize=9)


def add_boundary_legend(ax: plt.Axes, include_eastern_morocco: bool) -> None:
    """Add a compact line-style legend."""
    handles = []
    if include_eastern_morocco:
        handles.append(
            Line2D(
                [0],
                [0],
                color=EASTERN_MOROCCO_COLOR,
                linewidth=1.8,
                label="Eastern Morocco boundary",
            )
        )

    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=STUDY_AREA_COLOR,
                linewidth=2.0,
                label="Study-area boundary",
            ),
            Line2D(
                [0],
                [0],
                color=BUFFER_COLOR,
                linewidth=1.4,
                linestyle="--",
                label="10-km buffer",
            ),
        ]
    )

    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=8.5,
        frameon=True,
        framealpha=0.9,
    )


def render_single_frame(
    path: Path,
    region: str,
    year: int,
    boundaries: dict[str, gpd.GeoDataFrame],
) -> Image.Image:
    """Render one annual frame for one region with study boundaries."""
    data, extent = read_raster_for_display(path)
    label = REGIONS[region]

    fig = plt.figure(figsize=(8.5, 6.5), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.04, 0.07, 0.79, 0.83])
    cax = fig.add_axes([0.87, 0.18, 0.028, 0.63])

    ax.imshow(
        data,
        cmap=CMAP,
        norm=NORM,
        extent=extent,
        origin="upper",
        interpolation="nearest",
        zorder=1,
    )
    plot_region_boundaries(ax, boundaries, region)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        f"{label}\nAnnual predicted live vegetation cover - {year}",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )

    add_boundary_legend(ax, include_eastern_morocco=False)
    add_colorbar(fig, cax)
    return figure_to_pillow_image(fig)


def render_comparison_frame(
    al_baten_path: Path,
    bouarfa_path: Path,
    year: int,
    boundaries: dict[str, gpd.GeoDataFrame],
) -> Image.Image:
    """Render Al Baten and Bouarfa side by side with boundaries."""
    al_baten_data, al_baten_extent = read_raster_for_display(al_baten_path)
    bouarfa_data, bouarfa_extent = read_raster_for_display(bouarfa_path)

    fig = plt.figure(figsize=(13.5, 6.3), dpi=DPI, facecolor="white")
    ax1 = fig.add_axes([0.025, 0.08, 0.405, 0.78])
    ax2 = fig.add_axes([0.455, 0.08, 0.405, 0.78])
    cax = fig.add_axes([0.90, 0.18, 0.02, 0.60])

    for ax, data, extent, title, region in (
        (ax1, al_baten_data, al_baten_extent, "Al Baten", "Al_Baten"),
        (ax2, bouarfa_data, bouarfa_extent, "Bouarfa", "Bouarfa"),
    ):
        ax.imshow(
            data,
            cmap=CMAP,
            norm=NORM,
            extent=extent,
            origin="upper",
            interpolation="nearest",
            zorder=1,
        )
        plot_region_boundaries(ax, boundaries, region)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)

    add_boundary_legend(ax1, include_eastern_morocco=False)

    fig.suptitle(
        f"Annual predicted live vegetation cover - {year}",
        fontsize=17,
        fontweight="bold",
        y=0.95,
    )

    add_colorbar(fig, cax)
    return figure_to_pillow_image(fig)


def render_eastern_morocco_context_frame(
    al_baten_path: Path,
    bouarfa_path: Path,
    year: int,
    boundaries: dict[str, gpd.GeoDataFrame],
) -> Image.Image:
    """Render both regions in geographic context inside Eastern Morocco."""
    al_baten_data, al_baten_extent = read_raster_for_display(al_baten_path)
    bouarfa_data, bouarfa_extent = read_raster_for_display(bouarfa_path)

    fig = plt.figure(figsize=(12.8, 7.4), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.035, 0.07, 0.82, 0.84])
    cax = fig.add_axes([0.89, 0.18, 0.022, 0.62])

    ax.set_facecolor("#f7f7f7")

    # Plot prediction rasters in their actual map positions.
    ax.imshow(
        al_baten_data,
        cmap=CMAP,
        norm=NORM,
        extent=al_baten_extent,
        origin="upper",
        interpolation="nearest",
        zorder=2,
    )
    ax.imshow(
        bouarfa_data,
        cmap=CMAP,
        norm=NORM,
        extent=bouarfa_extent,
        origin="upper",
        interpolation="nearest",
        zorder=2,
    )

    # Eastern Morocco boundary.
    boundaries["Eastern_Morocco"].boundary.plot(
        ax=ax,
        linewidth=1.8,
        color=EASTERN_MOROCCO_COLOR,
        zorder=4,
    )

    # Regional study-area and buffer boundaries.
    plot_region_boundaries(ax, boundaries, "Al_Baten")
    plot_region_boundaries(ax, boundaries, "Bouarfa")

    add_region_label(ax, boundaries["Al_Baten"], "Al Baten")
    add_region_label(ax, boundaries["Bouarfa"], "Bouarfa")

    # Use the Eastern Morocco polygon to set a stable extent for every year.
    minx, miny, maxx, maxy = boundaries["Eastern_Morocco"].total_bounds
    xpad = (maxx - minx) * 0.035
    ypad = (maxy - miny) * 0.035
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        f"Eastern Morocco - Annual predicted live vegetation cover - {year}",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )

    add_boundary_legend(ax, include_eastern_morocco=True)
    add_colorbar(fig, cax)
    return figure_to_pillow_image(fig)


# =============================================================================
# 6. GIF CREATION
# =============================================================================


def save_gif(frames: list[Image.Image], output_path: Path) -> None:
    """Save Pillow images as a looping animated GIF."""
    if not frames:
        raise ValueError("Cannot save a GIF with no frames.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    durations = [FRAME_DURATION_MS] * len(frames)
    durations[-1] = LAST_FRAME_DURATION_MS

    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=False,
    )

    for frame in frames:
        frame.close()

    print(f"Created: {output_path}")


def make_individual_gif(
    region: str,
    files_by_year: dict[int, Path],
    years: list[int],
    boundaries: dict[str, gpd.GeoDataFrame],
) -> None:
    """Create one GIF for a single region."""
    if not years:
        return

    label = REGIONS[region]
    print(f"Rendering {label}: {years[0]}-{years[-1]}")

    frames = [
        render_single_frame(files_by_year[year], region, year, boundaries)
        for year in years
    ]

    output_path = OUTPUT_DIR / (
        f"Landsat_RF_{region}_F_alive_{years[0]}-{years[-1]}.gif"
    )
    save_gif(frames, output_path)


def make_side_by_side_gif(
    all_files: dict[str, dict[int, Path]],
    years_by_region: dict[str, list[int]],
    boundaries: dict[str, gpd.GeoDataFrame],
) -> None:
    """Create a synchronized Al Baten-Bouarfa comparison GIF."""
    common_years = sorted(
        set(years_by_region["Al_Baten"]).intersection(
            years_by_region["Bouarfa"]
        )
    )

    if not common_years:
        print("WARNING - No common Al Baten/Bouarfa years; comparison skipped.")
        return

    print(
        "Rendering side-by-side comparison: "
        f"{common_years[0]}-{common_years[-1]}"
    )

    frames = [
        render_comparison_frame(
            all_files["Al_Baten"][year],
            all_files["Bouarfa"][year],
            year,
            boundaries,
        )
        for year in common_years
    ]

    output_path = OUTPUT_DIR / (
        "Landsat_RF_Al_Baten_Bouarfa_"
        f"F_alive_{common_years[0]}-{common_years[-1]}.gif"
    )
    save_gif(frames, output_path)


def make_eastern_morocco_context_gif(
    all_files: dict[str, dict[int, Path]],
    years_by_region: dict[str, list[int]],
    boundaries: dict[str, gpd.GeoDataFrame],
) -> None:
    """Create one map-style GIF showing both regions inside Eastern Morocco."""
    common_years = sorted(
        set(years_by_region["Al_Baten"]).intersection(
            years_by_region["Bouarfa"]
        )
    )

    if not common_years:
        print("WARNING - No common years; Eastern Morocco context GIF skipped.")
        return

    print(
        "Rendering Eastern Morocco context map: "
        f"{common_years[0]}-{common_years[-1]}"
    )

    frames = [
        render_eastern_morocco_context_frame(
            all_files["Al_Baten"][year],
            all_files["Bouarfa"][year],
            year,
            boundaries,
        )
        for year in common_years
    ]

    output_path = OUTPUT_DIR / (
        "Landsat_RF_Eastern_Morocco_Context_"
        f"F_alive_{common_years[0]}-{common_years[-1]}.gif"
    )
    save_gif(frames, output_path)


# =============================================================================
# 7. RUN
# =============================================================================


def main() -> None:
    all_files = discover_rasters()

    years_by_region: dict[str, list[int]] = {}
    for region, label in REGIONS.items():
        years = selected_years(all_files[region], label)
        validate_region_grid(all_files[region], years, label)
        years_by_region[region] = years

    # Need at least one raster to determine the working CRS.
    first_raster = None
    for region in REGIONS:
        if years_by_region[region]:
            first_raster = all_files[region][years_by_region[region][0]]
            break

    if first_raster is None:
        raise FileNotFoundError("No prediction rasters were found.")

    raster_crs = get_raster_crs(first_raster)
    print(f"Raster CRS: {raster_crs}")

    raw_boundaries = load_boundaries()
    boundaries = reproject_boundaries(raw_boundaries, raster_crs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if MAKE_INDIVIDUAL_GIFS:
        for region in REGIONS:
            make_individual_gif(
                region,
                all_files[region],
                years_by_region[region],
                boundaries,
            )

    if MAKE_SIDE_BY_SIDE_GIF:
        make_side_by_side_gif(all_files, years_by_region, boundaries)

    if MAKE_EASTERN_MOROCCO_CONTEXT_GIF:
        make_eastern_morocco_context_gif(
            all_files,
            years_by_region,
            boundaries,
        )

    print("Finished.")


if __name__ == "__main__":
    main()