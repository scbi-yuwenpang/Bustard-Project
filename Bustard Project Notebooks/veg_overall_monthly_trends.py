"""
Overall monthly live-vegetation trend by selected satellite(s)
Creat on 9/3/2026
Purpose
-------
Create a communication-level overall vegetation-cover trend by:
1. Calculating the monthly mean within each habitat.
2. Averaging habitat-level monthly means so each habitat has equal weight.
3. Applying LOWESS smoothing.
4. Calculating an approximate LOWESS slope, pseudo-R2, and Mann-Kendall trend.
5. Exporting one figure per selected satellite and one summary CSV.

To run Sentinel-2 only, keep:
    SATELLITES_TO_RUN = ['Sentinel-2']

Other examples:
    SATELLITES_TO_RUN = ['Landsat']
    SATELLITES_TO_RUN = ['MODIS']
    SATELLITES_TO_RUN = ['Landsat', 'Sentinel-2', 'MODIS']
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from pymannkendall import original_test as mk_test


# ============================================================
# 1. USER SETTINGS
# ============================================================

BASE_DIR = Path(r"C:\Users\PangY\Python-GIS")

INPUT_DIR = BASE_DIR / "results" / "RF_Prediction"
OUTPUT_DIR = BASE_DIR / "plots" / "Plot_RF_Communication"

VALUE_COL = "F_alive_Pred"
HABITAT_COL = "Habitat_2_"
DATE_COL = "date"

LOWESS_FRAC = 0.30

# Select the satellite(s) you want to process.
# For Sentinel-2 only:
SATELLITES_TO_RUN = ["Sentinel-2"]

# Available files
SATELLITE_FILES = {
    "Landsat": "rf_pred_landsat_26.csv",
    "Sentinel-2": "rf_pred_sentinel2_26.csv",
    "MODIS": "rf_pred_modis_26.csv",
}


# ============================================================
# 2. LOAD SELECTED SATELLITE DATA
# ============================================================

def load_selected_satellites(
    input_dir,
    satellites_to_run,
    satellite_files
):
    """Load only the selected satellite CSV files."""

    satellite_map = {}

    for sat in satellites_to_run:

        if sat not in satellite_files:
            raise ValueError(
                f"Unknown satellite: {sat}\n"
                f"Available options: {list(satellite_files.keys())}"
            )

        csv_path = input_dir / satellite_files[sat]

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Input file not found:\n{csv_path}"
            )

        print(f"Loading {sat}: {csv_path}")

        satellite_map[sat] = pd.read_csv(csv_path)

    return satellite_map


# ============================================================
# 3. CALCULATE OVERALL MONTHLY TREND
# ============================================================

def calculate_overall_monthly_trend(
    df,
    value_col=VALUE_COL,
    habitat_col=HABITAT_COL,
    date_col=DATE_COL,
    frac=LOWESS_FRAC
):
    """
    Calculate the overall monthly vegetation trend.

    Important:
    The calculation is done in two stages:
        habitat -> monthly mean
        monthly habitat means -> overall monthly mean

    This gives each habitat equal weight and avoids habitats with
    larger sample sizes dominating the overall trend.
    """

    df = df.copy()

    # ---- Date conversion ----
    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    # ---- Remove missing values ----
    df = df.dropna(
        subset=[
            date_col,
            value_col,
            habitat_col
        ]
    )

    # ---- Monthly period ----
    df["month"] = df[date_col].dt.to_period("M")

    # --------------------------------------------------------
    # Step 1: mean within each habitat for each month
    # --------------------------------------------------------

    habitat_monthly = (
        df
        .groupby(
            ["month", habitat_col],
            as_index=False
        )[value_col]
        .mean()
    )

    # --------------------------------------------------------
    # Step 2: mean across habitat-level monthly means
    # --------------------------------------------------------

    monthly = (
        habitat_monthly
        .groupby(
            "month",
            as_index=False
        )
        .agg(
            mean_cover=(value_col, "mean"),
            n_habitats=(habitat_col, "nunique")
        )
    )

    monthly["date"] = monthly["month"].dt.to_timestamp()
    monthly = monthly.sort_values("date").reset_index(drop=True)

    if len(monthly) < 12:
        raise ValueError(
            "Fewer than 12 monthly observations are available."
        )

    # --------------------------------------------------------
    # LOWESS
    # --------------------------------------------------------

    x_days = (
        monthly["date"] - monthly["date"].min()
    ).dt.days.to_numpy()

    monthly["lowess"] = lowess(
        monthly["mean_cover"],
        x_days,
        frac=frac,
        return_sorted=False
    )

    # --------------------------------------------------------
    # Approximate LOWESS slope
    # --------------------------------------------------------

    time_span_years = (
        x_days[-1] - x_days[0]
    ) / 365.25

    slope = (
        monthly["lowess"].iloc[-1]
        - monthly["lowess"].iloc[0]
    ) / time_span_years

    # --------------------------------------------------------
    # Pseudo-R2
    # --------------------------------------------------------

    if (
        monthly["mean_cover"].std() > 0
        and monthly["lowess"].std() > 0
    ):
        r = np.corrcoef(
            monthly["mean_cover"],
            monthly["lowess"]
        )[0, 1]
        pseudo_r2 = r ** 2
    else:
        pseudo_r2 = np.nan

    # --------------------------------------------------------
    # Mann-Kendall
    # --------------------------------------------------------

    try:
        mk = mk_test(
            monthly["mean_cover"].to_numpy()
        )

        mk_tau = mk.Tau
        mk_p = mk.p
        mk_trend = mk.trend

    except Exception:
        mk_tau = np.nan
        mk_p = np.nan
        mk_trend = "no trend"

    stats = {
        "Slope_per_year": slope,
        "Slope_percentage_points_per_year": slope * 100,
        "Pseudo_R2": pseudo_r2,
        "MK_tau": mk_tau,
        "MK_p": mk_p,
        "MK_trend": mk_trend,
        "N_months": len(monthly),
        "Mean_habitats_per_month": monthly["n_habitats"].mean()
    }

    return monthly, stats


# ============================================================
# 4. PLOT ONE SATELLITE
# ============================================================

def plot_overall_trend(
    monthly,
    stats,
    satellite,
    outdir,
    frac=LOWESS_FRAC
):
    """Create and save one overall trend figure."""

    fig, ax = plt.subplots(
        figsize=(8.2, 4.5)
    )

    # ---- Monthly overall mean ----
    ax.scatter(
        monthly["date"],
        monthly["mean_cover"],
        s=38,
        color="gray",
        alpha=0.65,
        edgecolor="none",
        label="Monthly mean"
    )

    # ---- LOWESS curve ----
    ax.plot(
        monthly["date"],
        monthly["lowess"],
        lw=2.8,
        ls="--",
        color="steelblue",
        label=f"LOWESS (frac={frac})"
    )

    # ---- Statistics ----
    mk_p = stats["MK_p"]

    if np.isnan(mk_p):
        p_text = "NA"
    elif mk_p < 0.001:
        p_text = "< 0.001"
    else:
        p_text = f"= {mk_p:.3f}"

    stats_text = (
        f"Slope ≈ {stats['Slope_per_year']:.4f} / yr\n"
        f"Pseudo-R² = {stats['Pseudo_R2']:.2f}\n"
        f"MK τ = {stats['MK_tau']:.2f}, p {p_text}\n"
        f"Trend: {stats['MK_trend']}"
    )

    ax.text(
        0.025,
        0.96,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(
            facecolor="white",
            alpha=0.80,
            edgecolor="none",
            pad=4
        )
    )

    # ---- Figure formatting ----
    ax.set_title(
        f"Overall vegetation trend – {satellite}",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Date",
        fontsize=11
    )

    ax.set_ylabel(
        "Predicted live vegetation cover",
        fontsize=11
    )

    ax.grid(
        alpha=0.25,
        linestyle="--"
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=9
    )

    fig.tight_layout()

    # Safe name for output file
    satellite_file_name = (
        satellite
        .replace(" ", "_")
        .replace("-", "")
    )

    figure_path = (
        outdir
        / f"Overall_monthly_trend_{satellite_file_name}.png"
    )

    fig.savefig(
        figure_path,
        dpi=400,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved figure: {figure_path}")

    return figure_path


# ============================================================
# 5. MAIN WORKFLOW
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    satellite_map = load_selected_satellites(
        input_dir=INPUT_DIR,
        satellites_to_run=SATELLITES_TO_RUN,
        satellite_files=SATELLITE_FILES
    )

    summary_rows = []

    for satellite, df in satellite_map.items():

        print("\n" + "=" * 60)
        print(f"Processing: {satellite}")
        print("=" * 60)

        monthly, stats = calculate_overall_monthly_trend(
            df=df,
            value_col=VALUE_COL,
            habitat_col=HABITAT_COL,
            date_col=DATE_COL,
            frac=LOWESS_FRAC
        )

        # Add satellite name to summary
        summary_rows.append({
            "Satellite": satellite,
            **stats
        })

        # Save monthly values used in the figure
        satellite_file_name = (
            satellite
            .replace(" ", "_")
            .replace("-", "")
        )

        monthly_csv = (
            OUTPUT_DIR
            / f"Overall_monthly_series_{satellite_file_name}.csv"
        )

        monthly.to_csv(
            monthly_csv,
            index=False
        )

        print(f"Saved monthly series: {monthly_csv}")

        # Plot
        plot_overall_trend(
            monthly=monthly,
            stats=stats,
            satellite=satellite,
            outdir=OUTPUT_DIR,
            frac=LOWESS_FRAC
        )

        print(
            f"\n{satellite} results:"
            f"\n  Slope = {stats['Slope_per_year']:.5f} / yr"
            f"\n  Pseudo-R2 = {stats['Pseudo_R2']:.3f}"
            f"\n  MK tau = {stats['MK_tau']:.3f}"
            f"\n  MK p = {stats['MK_p']:.4f}"
            f"\n  Trend = {stats['MK_trend']}"
        )

    # --------------------------------------------------------
    # Export summary CSV
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "Overall_monthly_trend_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("Analysis completed.")
    print(f"Output directory:\n{OUTPUT_DIR}")
    print(f"Summary table:\n{summary_path}")
    print("=" * 60)


# ============================================================
# 6. RUN
# ============================================================

if __name__ == "__main__":
    main()
