
"""
climate_analysis.py

Reusable climate-analysis utilities for monthly climatology and wet/dry-year
classification using area-level climate summaries exported from Google Earth Engine.

Expected input columns
----------------------
dataset, spatial_level, region, date, temp_C, precip_mm

Typical dataset values:
- TerraClimate
- ERA5-Land

Typical spatial_level values:
- Regional
- Focal
"""
__version__ = "2.0"

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ============================================================
# Helpers
# ============================================================

DATASET_ALIASES = {
    "terra": "TerraClimate",
    "terraclimate": "TerraClimate",
    "tc": "TerraClimate",
    "era5": "ERA5-Land",
    "era5-land": "ERA5-Land",
    "era5_land": "ERA5-Land",
    "era5land": "ERA5-Land",
}

SPATIAL_ALIASES = {
    "regional": "Regional",
    "region": "Regional",
    "focal": "Focal",
}


def _safe_name(value):
    """Create a file-safe text label."""
    value = str(value).strip()
    value = re.sub(r"[^\w\-]+", "_", value)
    return value.strip("_")


def _normalize_dataset(dataset):
    key = str(dataset).strip().lower()
    return DATASET_ALIASES.get(key, dataset)


def _normalize_spatial_level(spatial_level):
    key = str(spatial_level).strip().lower()
    return SPATIAL_ALIASES.get(key, spatial_level)


def load_climate_data(climate_file):
    """
    Load and prepare the climate CSV.

    Required columns:
        dataset, spatial_level, region, date, temp_C, precip_mm
    """
    climate_file = Path(climate_file)
    df = pd.read_csv(climate_file)

    required = {
        "dataset",
        "spatial_level",
        "region",
        "date",
        "temp_C",
        "precip_mm",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {climate_file.name}: "
            f"{sorted(missing)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        n_bad = int(df["date"].isna().sum())
        raise ValueError(f"{n_bad} rows contain invalid dates.")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    return df


def filter_climate_data(
    df,
    dataset,
    spatial_level,
    analysis_start_year=None,
    analysis_end_year=None,
):
    """Filter the climate table by dataset, spatial scale, and optional years."""

    dataset = _normalize_dataset(dataset)
    spatial_level = _normalize_spatial_level(spatial_level)

    subset = df[
        (df["dataset"] == dataset)
        & (df["spatial_level"] == spatial_level)
    ].copy()

    if analysis_start_year is not None:
        subset = subset[subset["year"] >= analysis_start_year]

    if analysis_end_year is not None:
        subset = subset[subset["year"] <= analysis_end_year]

    if subset.empty:
        raise ValueError(
            f"No records found for dataset={dataset!r}, "
            f"spatial_level={spatial_level!r}."
        )

    return subset


# ============================================================
# Monthly climatology
# ============================================================

def calculate_monthly_climatology(df):
    """
    Calculate mean monthly precipitation and temperature for each region.
    """
    monthly = (
        df.groupby(["dataset", "spatial_level", "region", "month"], as_index=False)
        .agg(
            temp_C=("temp_C", "mean"),
            precip_mm=("precip_mm", "mean"),
            n_years=("year", "nunique"),
        )
        .sort_values(["region", "month"])
    )

    return monthly


def plot_monthly_climate(
    monthly_df,
    outdir,
    period_label=None,
    show=True,
):
    """
    Plot monthly climate for all regions in one figure.

    Style:
    - Precipitation: gray bars with black edges
    - Temperature: black line with circle markers
    - One subplot per region
    - Common precipitation y-axis across regions
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Month labels
    # --------------------------------------------------
    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    # --------------------------------------------------
    # Basic information
    # --------------------------------------------------
    regions = monthly_df["region"].dropna().unique()

    dataset = monthly_df["dataset"].iloc[0]
    spatial_level = monthly_df["spatial_level"].iloc[0]

    # --------------------------------------------------
    # Common precipitation y-axis
    # --------------------------------------------------
    precip_max = monthly_df["precip_mm"].max() * 1.10

    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(regions),
        figsize=(7 * len(regions), 5),
        sharey=False
    )

    if len(regions) == 1:
        axes = [axes]

    # --------------------------------------------------
    # Plot each region
    # --------------------------------------------------
    for ax, region in zip(axes, regions):

        df = (
            monthly_df[
                monthly_df["region"] == region
            ]
            .sort_values("month")
            .copy()
        )

        # ==============================================
        # Precipitation
        # ==============================================
        ax.bar(
            df["month"],
            df["precip_mm"],
            color="gray",
            edgecolor="black",
            alpha=0.7,
            width=0.7,
            zorder=1
        )

        ax.set_ylabel("Precipitation (mm)")
        ax.set_xlabel("Month")

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_labels)

        ax.set_ylim(0, precip_max)

        # ==============================================
        # Temperature
        # ==============================================
        ax2 = ax.twinx()

        # Put temperature axis above precipitation bars
        ax2.set_zorder(ax.get_zorder() + 1)
        ax2.patch.set_visible(False)

        ax2.plot(
            df["month"],
            df["temp_C"],
            color="black",
            marker="o",
            linewidth=2,
            zorder=10
        )

        ax2.set_ylabel("Temperature (°C)")

        # ==============================================
        # Panel title
        # ==============================================
        title = f"{region} - {dataset}"
        if period_label:
            title += f"\n{period_label}"
        ax.set_title(title)

    # --------------------------------------------------
    # Optional overall title
    # --------------------------------------------------
    if period_label is not None:
        fig.suptitle(
            f"{period_label}",
            fontsize=12,
            y=1.02
        )

    plt.tight_layout()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    outfile = outdir / (
        f"Monthly_Climate_"
        f"{_safe_name(dataset)}_"
        f"{_safe_name(spatial_level)}.png"
    )

    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {outfile}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return [outfile]


# ============================================================
# Annual / ecological-year climate
# ============================================================

def calculate_annual_climate(
    df,
    year_type="calendar",
    eco_start_month=9,
    require_complete_year=True,
):
    """
    Aggregate monthly climate to annual values.

    Parameters
    ----------
    year_type : {"calendar", "ecological"}
        calendar   = January-December
        ecological = eco_start_month through the month before eco_start_month
                     in the following calendar year.

    eco_start_month : int
        Default = 9, so ecological year = September-August.

    Notes
    -----
    An ecological year beginning in Sep 2020 and ending in Aug 2021
    is labeled 2021.
    """
    annual_source = df.copy()

    year_type = str(year_type).strip().lower()

    if year_type == "calendar":
        annual_source["analysis_year"] = annual_source["year"]

    elif year_type == "ecological":
        annual_source["analysis_year"] = np.where(
            annual_source["month"] >= eco_start_month,
            annual_source["year"] + 1,
            annual_source["year"],
        )

    else:
        raise ValueError("year_type must be 'calendar' or 'ecological'.")

    annual = (
        annual_source.groupby(
            ["dataset", "spatial_level", "region", "analysis_year"],
            as_index=False,
        )
        .agg(
            annual_precip_mm=("precip_mm", "sum"),
            annual_temp_C=("temp_C", "mean"),
            n_months=("month", "nunique"),
        )
        .sort_values(["region", "analysis_year"])
    )

    if require_complete_year:
        annual = annual[annual["n_months"] == 12].copy()

    annual["year_type"] = year_type

    return annual


# ============================================================
# Wet / dry classification
# ============================================================

def classify_wet_dry_years(
    annual_df,
    baseline_start_year=1991,
    baseline_end_year=2020,
    dry_quantile=0.25,
    wet_quantile=0.75,
):
    """
    Classify dry / normal / wet years relative to a fixed baseline.
    WMO defines climate normals using 30-year periods, with 1991–2020 as the current standard normal
    
    Classification
    --------------
    Dry:
        annual precipitation <= baseline dry_quantile

    Wet:
        annual precipitation >= baseline wet_quantile

    Normal:
        between the two thresholds

    The baseline mean is also used to calculate precipitation anomalies.
    """

    if baseline_end_year < baseline_start_year:
        raise ValueError("baseline_end_year must be >= baseline_start_year.")

    if not 0 < dry_quantile < wet_quantile < 1:
        raise ValueError(
            "Quantiles must satisfy 0 < dry_quantile < wet_quantile < 1."
        )

    df = annual_df.copy()

    baseline = df[
        (df["analysis_year"] >= baseline_start_year)
        & (df["analysis_year"] <= baseline_end_year)
    ].copy()

    if baseline.empty:
        raise ValueError(
            f"No complete annual records found in baseline "
            f"{baseline_start_year}-{baseline_end_year}."
        )

    baseline_stats = (
        baseline.groupby(
            ["dataset", "spatial_level", "region"],
            as_index=False,
        )
        .agg(
            baseline_n_years=("analysis_year", "nunique"),
            baseline_mean_precip_mm=("annual_precip_mm", "mean"),
            baseline_median_precip_mm=("annual_precip_mm", "median"),
            baseline_std_precip_mm=("annual_precip_mm", "std"),
            dry_threshold_mm=(
                "annual_precip_mm",
                lambda x: x.quantile(dry_quantile),
            ),
            wet_threshold_mm=(
                "annual_precip_mm",
                lambda x: x.quantile(wet_quantile),
            ),
        )
    )

    expected_years = baseline_end_year - baseline_start_year + 1

    incomplete = baseline_stats[
        baseline_stats["baseline_n_years"] < expected_years
    ]

    if not incomplete.empty:
        warnings.warn(
            "Some regions do not contain the full requested baseline period. "
            "Check baseline_n_years in the baseline summary table."
        )

    classified = df.merge(
        baseline_stats,
        on=["dataset", "spatial_level", "region"],
        how="left",
    )

    classified["precip_anomaly_mm"] = (
        classified["annual_precip_mm"]
        - classified["baseline_mean_precip_mm"]
    )

    classified["precip_anomaly_pct"] = (
        classified["precip_anomaly_mm"]
        / classified["baseline_mean_precip_mm"]
        * 100
    )

    classified["precip_zscore"] = (
        classified["annual_precip_mm"]
        - classified["baseline_mean_precip_mm"]
    ) / classified["baseline_std_precip_mm"]

    classified["climate_class"] = np.select(
        [
            classified["annual_precip_mm"]
            <= classified["dry_threshold_mm"],

            classified["annual_precip_mm"]
            >= classified["wet_threshold_mm"],
        ],
        [
            "Dry",
            "Wet",
        ],
        default="Normal",
    )

    classified["baseline_start_year"] = baseline_start_year
    classified["baseline_end_year"] = baseline_end_year

    baseline_stats["baseline_start_year"] = baseline_start_year
    baseline_stats["baseline_end_year"] = baseline_end_year
    baseline_stats["dry_quantile"] = dry_quantile
    baseline_stats["wet_quantile"] = wet_quantile

    return classified, baseline_stats


def make_wet_dry_summary(classified_df):
    """
    Create a compact table listing dry and wet years for each region.
    """

    subset = classified_df[
        classified_df["climate_class"].isin(["Dry", "Wet"])
    ].copy()

    if subset.empty:
        return pd.DataFrame(
            columns=["dataset", "spatial_level", "region", "Dry", "Wet"]
        )

    summary = (
        subset.groupby(
            ["dataset", "spatial_level", "region", "climate_class"]
        )["analysis_year"]
        .apply(lambda x: ", ".join(map(str, sorted(x.astype(int).tolist()))))
        .unstack(fill_value="")
        .reset_index()
    )

    if "Dry" not in summary.columns:
        summary["Dry"] = ""

    if "Wet" not in summary.columns:
        summary["Wet"] = ""

    return summary[
        ["dataset", "spatial_level", "region", "Dry", "Wet"]
    ]


# ============================================================
# Wet / dry anomaly figure
# ============================================================

def plot_wet_dry_anomalies(
    classified_df,
    outdir,
    show=True,
):
    """
    Plot annual precipitation and Dry / Normal / Wet classification.

    Publication-style format:
    - absolute annual precipitation
    - Dry / Normal / Wet shown with muted colors
    - baseline shown as a dashed horizontal line
    - baseline precipitation explicitly added to the y-axis
    - baseline period labeled directly on the dashed line
    """

    from matplotlib.patches import Patch

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_files = []

    # Muted journal-style palette
    class_colors = {
        "Dry": "#C8C8C8",
        "Normal": "#8FC1DD",
        "Wet": "#4A86D9",
    }

    for region in classified_df["region"].dropna().unique():

        df = (
            classified_df[
                classified_df["region"] == region
            ]
            .sort_values("analysis_year")
            .copy()
        )

        dataset = df["dataset"].iloc[0]
        spatial_level = df["spatial_level"].iloc[0]
        year_type = df["year_type"].iloc[0]

        baseline_mean = df["baseline_mean_precip_mm"].iloc[0]

        baseline_start = int(
            df["baseline_start_year"].iloc[0]
        )

        baseline_end = int(
            df["baseline_end_year"].iloc[0]
        )

        # --------------------------------------------------
        # Colors
        # --------------------------------------------------
        bar_colors = (
            df["climate_class"]
            .map(class_colors)
            .fillna(class_colors["Normal"])
        )

        # --------------------------------------------------
        # Figure
        # --------------------------------------------------
        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        # Annual precipitation
        ax.bar(
            df["analysis_year"],
            df["annual_precip_mm"],
            color=bar_colors,
            edgecolor="black",
            linewidth=0.65,
            width=0.78,
            zorder=3
        )

        # --------------------------------------------------
        # Baseline
        # --------------------------------------------------
        ax.axhline(
            baseline_mean,
            color="#C43C35",
            linestyle="--",
            linewidth=1.6,
            zorder=4
        )

        # Put baseline VALUE explicitly on y-axis
        current_ticks = list(ax.get_yticks())

        if not any(
            abs(t - baseline_mean) < 0.01
            for t in current_ticks
        ):
            current_ticks.append(baseline_mean)

        current_ticks = sorted(current_ticks)

        ax.set_yticks(current_ticks)

        # Format labels
        tick_labels = []

        for tick in current_ticks:
            if abs(tick - baseline_mean) < 0.01:
                tick_labels.append(
                    f"{baseline_mean:.1f}"
                )
            else:
                tick_labels.append(
                    f"{tick:.0f}"
                )

        ax.set_yticklabels(tick_labels)

        # Highlight baseline tick label
        for label, tick in zip(
            ax.get_yticklabels(),
            current_ticks
        ):
            if abs(tick - baseline_mean) < 0.01:
                label.set_color("#C43C35")
                label.set_fontweight("bold")

        # --------------------------------------------------
        # Baseline label directly on line
        # --------------------------------------------------
        ax.text(
            0.68,
            baseline_mean,
            f"  {baseline_start}–{baseline_end} baseline  ",
            transform=ax.get_yaxis_transform(),
            color="#C43C35",
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
                pad=1
            ),
            zorder=6
        )

        # --------------------------------------------------
        # Axes
        # --------------------------------------------------
        ax.set_ylabel(
            "Annual precipitation (mm)"
        )

        if year_type == "calendar":
            ax.set_xlabel("Year")
        else:
            ax.set_xlabel("Ecological year")

        # Show every second year
        years = df["analysis_year"].astype(int)

        tick_years = years.iloc[::2]

        ax.set_xticks(tick_years)

        ax.set_xticklabels(
            tick_years,
            rotation=45,
            ha="right"
        )

        ax.set_ylim(
            0,
            df["annual_precip_mm"].max() * 1.12
        )

        # --------------------------------------------------
        # Journal-style formatting
        # --------------------------------------------------
        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.6,
            alpha=0.25,
            zorder=0
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(
            axis="both",
            direction="out"
        )

        # --------------------------------------------------
        # Title
        # --------------------------------------------------
        ax.set_title(
            f"{region} – {dataset}",
            fontsize=12,
            pad=10
        )

        # --------------------------------------------------
        # Legend: climate classes only
        # --------------------------------------------------
        legend_handles = [
            Patch(
                facecolor=class_colors["Dry"],
                edgecolor="black",
                label="Dry"
            ),
            Patch(
                facecolor=class_colors["Normal"],
                edgecolor="black",
                label="Normal"
            ),
            Patch(
                facecolor=class_colors["Wet"],
                edgecolor="black",
                label="Wet"
            )
        ]

        ax.legend(
            handles=legend_handles,
            frameon=False,
            loc="upper right",
            ncol=3
        )

        fig.tight_layout()

        # --------------------------------------------------
        # Save
        # --------------------------------------------------
        outfile = outdir / (
            f"Wet_Dry_Annual_Precipitation_"
            f"{_safe_name(dataset)}_"
            f"{_safe_name(spatial_level)}_"
            f"{_safe_name(region)}_"
            f"{_safe_name(year_type)}.png"
        )

        fig.savefig(
            outfile,
            dpi=300,
            bbox_inches="tight"
        )

        output_files.append(outfile)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return output_files


# ============================================================
# Main wrapper
# ============================================================

def run_climate_analysis(
    climate_file,
    outdir,
    spatial_scale,
    climate_source,
    baseline_start_year,
    baseline_end_year,
    analysis_start_year=None,
    analysis_end_year=None,
    year_type="calendar",
    eco_start_month=9,
    dry_quantile=0.25,
    wet_quantile=0.75,
    show_plots=True,
):
    """
    Complete climate-analysis workflow.

    Parameters
    ----------
    climate_file : str or Path
        Input monthly climate CSV.

    outdir : str or Path
        Root output folder.

    spatial_scale : str
        "Regional" or "Focal".

    climate_source : str
        "TerraClimate" or "ERA5-Land".
        Aliases such as "Terra" and "ERA5" are also accepted.

    baseline_start_year, baseline_end_year : int
        Fixed baseline period used for wet / dry classification.

    analysis_start_year, analysis_end_year : int or None
        Optional period to include in outputs.
        If None, all available data are used.

    year_type : {"calendar", "ecological"}
        Annual aggregation method.

    eco_start_month : int
        Start month for ecological year. Default = 9 (September).

    dry_quantile, wet_quantile : float
        Baseline precipitation thresholds for dry and wet classification.

    Returns
    -------
    dict
        DataFrames and output-file paths.
    """

    climate_source = _normalize_dataset(climate_source)
    spatial_scale = _normalize_spatial_level(spatial_scale)

    outdir = Path(outdir)

    table_dir = outdir / "tables"
    figure_dir = outdir / "figures"

    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load + filter
    # --------------------------------------------------------
    climate = load_climate_data(climate_file)

    subset = filter_climate_data(
        climate,
        dataset=climate_source,
        spatial_level=spatial_scale,
        analysis_start_year=analysis_start_year,
        analysis_end_year=analysis_end_year,
    )

    actual_start = int(subset["year"].min())
    actual_end = int(subset["year"].max())

    period_label = f"{actual_start}-{actual_end}"

    prefix = (
        f"{_safe_name(climate_source)}_"
        f"{_safe_name(spatial_scale)}"
    )

    # --------------------------------------------------------
    # 2. Monthly climatology
    # --------------------------------------------------------
    monthly = calculate_monthly_climatology(subset)

    monthly_file = table_dir / (
        f"{prefix}_Monthly_Climatology.csv"
    )

    monthly.to_csv(monthly_file, index=False)

    monthly_figures = plot_monthly_climate(
        monthly,
        outdir=figure_dir,
        period_label=period_label,
        show=show_plots,
    )

    # --------------------------------------------------------
    # 3. Annual climate
    # --------------------------------------------------------
    annual = calculate_annual_climate(
        subset,
        year_type=year_type,
        eco_start_month=eco_start_month,
        require_complete_year=True,
    )

    # --------------------------------------------------------
    # 4. Wet / dry classification
    # --------------------------------------------------------
    classified, baseline_stats = classify_wet_dry_years(
        annual,
        baseline_start_year=baseline_start_year,
        baseline_end_year=baseline_end_year,
        dry_quantile=dry_quantile,
        wet_quantile=wet_quantile,
    )

    annual_file = table_dir / (
        f"{prefix}_{_safe_name(year_type)}_"
        f"Wet_Dry_Annual_Summary.csv"
    )

    baseline_file = table_dir / (
        f"{prefix}_{_safe_name(year_type)}_"
        f"Baseline_Summary.csv"
    )

    classified.to_csv(annual_file, index=False)
    baseline_stats.to_csv(baseline_file, index=False)

    # --------------------------------------------------------
    # 5. Compact wet / dry year summary
    # --------------------------------------------------------
    wet_dry_summary = make_wet_dry_summary(classified)

    wet_dry_summary_file = table_dir / (
        f"{prefix}_{_safe_name(year_type)}_"
        f"Wet_Dry_Years.csv"
    )

    wet_dry_summary.to_csv(
        wet_dry_summary_file,
        index=False,
    )

    # --------------------------------------------------------
    # 6. Wet / dry anomaly figures
    # --------------------------------------------------------
    wet_dry_figures = plot_wet_dry_anomalies(
        classified,
        outdir=figure_dir,
        show=show_plots,
    )

    # --------------------------------------------------------
    # 7. Print concise summary
    # --------------------------------------------------------
    print("=" * 70)
    print("Climate analysis completed")
    print("=" * 70)
    print(f"Dataset:        {climate_source}")
    print(f"Spatial scale:  {spatial_scale}")
    print(f"Analysis range: {actual_start}-{actual_end}")
    print(
        f"Baseline:       "
        f"{baseline_start_year}-{baseline_end_year}"
    )
    print(f"Year type:      {year_type}")
    print()

    print("Baseline precipitation summary:")
    print(
        baseline_stats[
            [
                "region",
                "baseline_n_years",
                "baseline_mean_precip_mm",
                "dry_threshold_mm",
                "wet_threshold_mm",
            ]
        ].round(1).to_string(index=False)
    )

    print("\nDry and wet years:")
    if wet_dry_summary.empty:
        print("No dry/wet years identified.")
    else:
        print(wet_dry_summary.to_string(index=False))

    print("\nTables:")
    print(f"  {monthly_file}")
    print(f"  {annual_file}")
    print(f"  {baseline_file}")
    print(f"  {wet_dry_summary_file}")

    print("\nFigures:")
    for file in monthly_figures + wet_dry_figures:
        print(f"  {file}")

    # --------------------------------------------------------
    # Return everything for notebook use
    # --------------------------------------------------------
    return {
        "filtered_data": subset,
        "monthly_climatology": monthly,
        "annual_climate": annual,
        "wet_dry_classification": classified,
        "baseline_summary": baseline_stats,
        "wet_dry_years": wet_dry_summary,
        "monthly_table": monthly_file,
        "annual_table": annual_file,
        "baseline_table": baseline_file,
        "wet_dry_summary_table": wet_dry_summary_file,
        "monthly_figures": monthly_figures,
        "wet_dry_figures": wet_dry_figures,
    }


# ============================================================
# Example
# ============================================================
#
# from pathlib import Path
#
# BASE_DIR = Path(r"C:\Users\PangY\Python-GIS")
#
# CLIMATE_FILE = (
#     BASE_DIR
#     / "csvs"
#     / "All_Climate_Areas_Monthly_2000_2025.csv"
# )
#
# OUTDIR = BASE_DIR / "plot" / "Plot_Clim"
#
# results = run_climate_analysis(
#     climate_file=CLIMATE_FILE,
#     outdir=OUTDIR,
#     spatial_scale="Regional",
#     climate_source="TerraClimate",
#     baseline_start_year=2000,
#     baseline_end_year=2014,
#     year_type="calendar",
# )
#
