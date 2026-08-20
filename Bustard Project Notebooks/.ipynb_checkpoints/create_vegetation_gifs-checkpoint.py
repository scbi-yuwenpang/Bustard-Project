from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Render without opening a GUI window.
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

# Folder containing the downloaded GEE GeoTIFFs.
# Windows example: Path(r"G:\My Drive\GEE_Landsat_Annual_Export")
# macOS/Linux example: Path("/home/name/GEE_Landsat_Annual_Export")
INPUT_DIR = Path(r"D:\GEE_Landsat_Annual_Export")

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

# If True, stop when any year from START_YEAR to END_YEAR is missing.
# If False, create the GIF from the available years and print a warning.
REQUIRE_ALL_YEARS = False

# Used only if the TIFF does not contain NoData metadata.
NODATA_FALLBACK = -9999.0

REGIONS = {
    "Al_Baten": "Al Baten",
    "Bouarfa": "Bouarfa",
}

# Same palette used in the GEE script.
GEE_PALETTE = [
    "#313695",
    "#4575b4",
    "#74add1",
    "#abd9e9",
    "#e0f3f8",
    "#ffffbf",
    "#fee090",
    "#fdae61",
    "#f46d43",
    "#d73027",
    "#a50026",
]

CMAP = LinearSegmentedColormap.from_list(
    "gee_live_vegetation_cover",
    GEE_PALETTE,
    N=256,
)
CMAP.set_bad("#eeeeee")  # NoData background.
NORM = Normalize(vmin=0.0, vmax=1.0, clip=True)

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


# =============================================================================
# 3. RASTER READING
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

        # The GEE prediction was clamped to 0-1. Values outside this range are
        # therefore treated as NoData or edge-resampling artifacts.
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
# 4. FRAME RENDERING
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
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    colorbar.set_label(
        "Predicted fractional live vegetation cover (F_alive)",
        fontsize=10,
        labelpad=10,
    )
    colorbar.ax.tick_params(labelsize=9)


def render_single_frame(path: Path, region_label: str, year: int) -> Image.Image:
    """Render one annual frame for one region."""
    data, extent = read_raster_for_display(path)

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
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        f"{region_label}\nAnnual predicted live vegetation cover - {year}",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )

    add_colorbar(fig, cax)
    return figure_to_pillow_image(fig)


def render_comparison_frame(
    al_baten_path: Path,
    bouarfa_path: Path,
    year: int,
) -> Image.Image:
    """Render Al Baten and Bouarfa side by side for one year."""
    al_baten_data, al_baten_extent = read_raster_for_display(al_baten_path)
    bouarfa_data, bouarfa_extent = read_raster_for_display(bouarfa_path)

    fig = plt.figure(figsize=(13.5, 6.3), dpi=DPI, facecolor="white")
    ax1 = fig.add_axes([0.025, 0.08, 0.405, 0.78])
    ax2 = fig.add_axes([0.455, 0.08, 0.405, 0.78])
    cax = fig.add_axes([0.90, 0.18, 0.02, 0.60])

    for ax, data, extent, title in (
        (ax1, al_baten_data, al_baten_extent, "Al Baten"),
        (ax2, bouarfa_data, bouarfa_extent, "Bouarfa"),
    ):
        ax.imshow(
            data,
            cmap=CMAP,
            norm=NORM,
            extent=extent,
            origin="upper",
            interpolation="nearest",
        )
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)

    fig.suptitle(
        f"Annual predicted live vegetation cover - {year}",
        fontsize=17,
        fontweight="bold",
        y=0.95,
    )

    add_colorbar(fig, cax)
    return figure_to_pillow_image(fig)


# =============================================================================
# 5. GIF CREATION
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
) -> None:
    """Create one GIF for a single region."""
    if not years:
        return

    label = REGIONS[region]
    print(f"Rendering {label}: {years[0]}-{years[-1]}")

    frames = [
        render_single_frame(files_by_year[year], label, year)
        for year in years
    ]

    output_path = OUTPUT_DIR / (
        f"Landsat_RF_{region}_F_alive_{years[0]}-{years[-1]}.gif"
    )
    save_gif(frames, output_path)


def make_side_by_side_gif(
    all_files: dict[str, dict[int, Path]],
    years_by_region: dict[str, list[int]],
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
        )
        for year in common_years
    ]

    output_path = OUTPUT_DIR / (
        "Landsat_RF_Al_Baten_Bouarfa_"
        f"F_alive_{common_years[0]}-{common_years[-1]}.gif"
    )
    save_gif(frames, output_path)


# =============================================================================
# 6. RUN
# =============================================================================


def main() -> None:
    all_files = discover_rasters()

    years_by_region: dict[str, list[int]] = {}
    for region, label in REGIONS.items():
        years = selected_years(all_files[region], label)
        validate_region_grid(all_files[region], years, label)
        years_by_region[region] = years

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if MAKE_INDIVIDUAL_GIFS:
        for region in REGIONS:
            make_individual_gif(
                region,
                all_files[region],
                years_by_region[region],
            )

    if MAKE_SIDE_BY_SIDE_GIF:
        make_side_by_side_gif(all_files, years_by_region)

    print("Finished.")


if __name__ == "__main__":
    main()
