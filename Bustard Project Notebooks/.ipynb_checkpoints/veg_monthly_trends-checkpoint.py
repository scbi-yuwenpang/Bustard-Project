"""
Create two communication trend products for selected satellite dataset(s):

1) OVERALL TREND
   - One overall monthly trajectory across all monitored habitat types.
   - Monthly means are first calculated within habitat, then averaged
     across habitats with equal weight.

2) HYDROECOLOGICAL-SEPARATED TRENDS
   - Steppe habitats combined:
       Reg
       Chamaephyte steppe
       Anabasis steppe
       Alfah-grass steppe
       Salty steppe
   - Hydrologically influenced habitats combined:
       Spreading area
       Wadi and gullies
   - Displayed as TWO PANELS in one figure.

This version intentionally follows the structure and plotting style of the
previous working habitat-trend code: gray monthly points + dashed steelblue
LOWESS + Mann-Kendall statistics.

"""

import os
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
DATE_COL = "date"
HABITAT_COL = "Habitat_2_"

LOWESS_FRAC = 0.30

# ------------------------------------------------------------
# Select target satellite dataset(s)
# ------------------------------------------------------------
# Sentinel-2 only:
SATELLITES_TO_RUN = ["Sentinel-2"]

# Other examples:
# SATELLITES_TO_RUN = ["Landsat"]
# SATELLITES_TO_RUN = ["MODIS"]
# SATELLITES_TO_RUN = ["Landsat", "Sentinel-2", "MODIS"]

SATELLITE_FILES = {
    "Landsat": "rf_pred_landsat_26.csv",
    "Sentinel-2": "rf_pred_sentinel2_26.csv",
    "MODIS": "rf_pred_modis_26.csv",
}


# ============================================================
# 2. HYDROECOLOGICAL GROUPS
# ============================================================

STEPPE_HABITATS = [
    "Reg",
    "Chamaephyte steppe",
    "Anabasis steppe",
    "Alfah-grass steppe",
    "Salty steppe",
]

HYDRO_HABITATS = [
    "Spreading area",
    "Wadi and gullies",
]


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_satellite_data(satellite):
    """
    Load one selected satellite CSV.
    """

    if satellite not in SATELLITE_FILES:
        raise ValueError(
            f"Unknown satellite: {satellite}\n"
            f"Choose from: {list(SATELLITE_FILES.keys())}"
        )

    csv_path = INPUT_DIR / SATELLITE_FILES[satellite]

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find input file:\n{csv_path}"
        )

    print(f"\nLoading {satellite}")
    print(f"  {csv_path}")

    df = pd.read_csv(csv_path)

    # Clean basic columns
    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce"
    )

    df[HABITAT_COL] = (
        df[HABITAT_COL]
        .astype(str)
        .str.strip()
    )

    df[VALUE_COL] = pd.to_numeric(
        df[VALUE_COL],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            DATE_COL,
            VALUE_COL,
            HABITAT_COL
        ]
    )

    print("  Habitat labels found:")
    for h in sorted(df[HABITAT_COL].unique()):
        print(f"    - {h}")

    return df


# ============================================================
# 4. MONTHLY HABITAT MEANS
# ============================================================

def make_monthly_habitat_means(df):
    """
    First aggregate predictions within each habitat and month.

    This is the key step that prevents habitats with more raw records
    from dominating the combined trend.
    """

    d = df.copy()

    d["month"] = d[DATE_COL].dt.to_period("M")

    habitat_monthly = (
        d
        .groupby(
            ["month", HABITAT_COL]
        )[VALUE_COL]
        .mean()
        .reset_index()
    )

    habitat_monthly["date"] = (
        habitat_monthly["month"]
        .dt.to_timestamp()
    )

    return habitat_monthly


# ============================================================
# 5. BUILD A COMBINED MONTHLY SERIES
# ============================================================

def combine_habitats_monthly(
    habitat_monthly,
    selected_habitats=None
):
    """
    Create an equal-weight monthly mean across selected habitat types.

    If selected_habitats is None:
        use all habitats.

    Otherwise:
        keep only the requested habitat group.
    """

    d = habitat_monthly.copy()

    if selected_habitats is not None:
        d = d[
            d[HABITAT_COL].isin(selected_habitats)
        ].copy()

    if d.empty:
        raise ValueError(
            "No data remained after habitat filtering. "
            "Check the habitat names printed when the script starts."
        )

    monthly = (
        d
        .groupby("month")
        .agg(
            mean_cover=(VALUE_COL, "mean"),
            n_habitats=(HABITAT_COL, "nunique")
        )
        .reset_index()
    )

    monthly["date"] = (
        monthly["month"]
        .dt.to_timestamp()
    )

    monthly = (
        monthly
        .sort_values("date")
        .reset_index(drop=True)
    )

    return monthly


# ============================================================
# 6. LOWESS + TREND STATISTICS
# ============================================================

def add_trend_statistics(
    monthly,
    frac=LOWESS_FRAC
):
    """
    Apply LOWESS and calculate the same style of statistics as the
    previous working plotting code.
    """

    d = monthly.copy()

    if len(d) < 12:
        raise ValueError(
            f"Only {len(d)} monthly observations found; "
            "at least 12 are required."
        )

    x_days = (
        d["date"] - d["date"].min()
    ).dt.days.values

    d["lowess"] = lowess(
        d["mean_cover"],
        x_days,
        frac=frac,
        return_sorted=False
    )

    # Approximate LOWESS endpoint slope per year
    total_days = x_days[-1] - x_days[0]

    if total_days == 0:
        slope = np.nan
    else:
        slope = (
            (
                d["lowess"].iloc[-1]
                - d["lowess"].iloc[0]
            )
            / total_days
        ) * 365.25

    # Pseudo-R2
    try:
        r2 = (
            np.corrcoef(
                d["mean_cover"],
                d["lowess"]
            )[0, 1] ** 2
        )
    except Exception:
        r2 = np.nan

    # Original Mann-Kendall, matching previous code
    try:
        mk = mk_test(
            d["mean_cover"].values
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
        "Pseudo_R2": r2,
        "MK_tau": mk_tau,
        "MK_p": mk_p,
        "MK_trend": mk_trend,
        "N_months": len(d),
        "Mean_habitats_per_month": d["n_habitats"].mean(),
        "Min_habitats_per_month": d["n_habitats"].min(),
        "Max_habitats_per_month": d["n_habitats"].max(),
    }

    return d, stats


# ============================================================
# 7. COMMON PANEL PLOTTING FUNCTION
# ============================================================

def draw_trend_panel(
    ax,
    monthly,
    stats,
    title,
    frac=LOWESS_FRAC
):
    """
    Draw one trend panel using the previous figure style.
    """

    ax.scatter(
        monthly["date"],
        monthly["mean_cover"],
        s=35,
        color="gray",
        alpha=0.65,
        label="Monthly mean"
    )

    ax.plot(
        monthly["date"],
        monthly["lowess"],
        lw=2.7,
        ls="--",
        color="steelblue",
        label=f"LOWESS (frac={frac})"
    )

    ax.set_title(
        title,
        fontsize=14
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
        alpha=0.30,
        linestyle="--"
    )

    ax.legend(
        loc="upper right",
        fontsize=9,
        frameon=True
    )

    p = stats["MK_p"]

    if pd.isna(p):
        p_text = "NA"
    elif p < 0.001:
        p_text = "< 0.001"
    else:
        p_text = f"= {p:.3f}"

    stats_text = (
        f"Slope ≈ {stats['Slope_per_year']:.4f} / yr\n"
        f"Pseudo-R² = {stats['Pseudo_R2']:.2f}\n"
        f"MK τ = {stats['MK_tau']:.2f}, p {p_text}\n"
        f"Trend: {stats['MK_trend']}"
    )

    ax.text(
        0.02,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(
            facecolor="white",
            alpha=0.75,
            edgecolor="none"
        )
    )


# ============================================================
# 8. FIGURE TYPE 1: OVERALL TREND
# ============================================================

def plot_overall_trend(
    habitat_monthly,
    satellite,
    outdir
):
    """
    Create one overall figure across all monitored habitat types.
    """

    monthly = combine_habitats_monthly(
        habitat_monthly,
        selected_habitats=None
    )

    monthly, stats = add_trend_statistics(
        monthly,
        frac=LOWESS_FRAC
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 4.6)
    )

    draw_trend_panel(
        ax=ax,
        monthly=monthly,
        stats=stats,
        title=f"Overall vegetation trend – {satellite}",
        frac=LOWESS_FRAC
    )

    fig.tight_layout()

    safe_sat = (
        satellite
        .replace(" ", "_")
        .replace("-", "")
    )

    fig_path = (
        outdir
        / f"Overall_trend_{safe_sat}.png"
    )

    fig.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    series_path = (
        outdir
        / f"Overall_monthly_series_{safe_sat}.csv"
    )

    monthly.to_csv(
        series_path,
        index=False
    )

    print(f"  Saved overall figure: {fig_path}")

    return stats


# ============================================================
# 9. FIGURE TYPE 2: HYDROECOLOGICAL-SEPARATED TRENDS
# ============================================================

def plot_hydroecological_trends(
    habitat_monthly,
    satellite,
    outdir
):
    """
    Create ONE figure with TWO panels:
        A. Steppe habitats combined
        B. Spreading area + Wadi and gullies combined
    """

    # --------------------------------------------------------
    # A. STEPPE GROUP
    # --------------------------------------------------------

    steppe_monthly = combine_habitats_monthly(
        habitat_monthly,
        selected_habitats=STEPPE_HABITATS
    )

    steppe_monthly, steppe_stats = (
        add_trend_statistics(
            steppe_monthly,
            frac=LOWESS_FRAC
        )
    )

    # --------------------------------------------------------
    # B. HYDROLOGICAL GROUP
    # --------------------------------------------------------

    hydro_monthly = combine_habitats_monthly(
        habitat_monthly,
        selected_habitats=HYDRO_HABITATS
    )

    hydro_monthly, hydro_stats = (
        add_trend_statistics(
            hydro_monthly,
            frac=LOWESS_FRAC
        )
    )

    # --------------------------------------------------------
    # Two-panel figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 4.4),
        sharey=False
    )

    draw_trend_panel(
        ax=axes[0],
        monthly=steppe_monthly,
        stats=steppe_stats,
        title=f"Steppe habitats – {satellite}",
        frac=LOWESS_FRAC
    )

    draw_trend_panel(
        ax=axes[1],
        monthly=hydro_monthly,
        stats=hydro_stats,
        title=(
            f"Spreading + wadi/gullies – {satellite}"
        ),
        frac=LOWESS_FRAC
    )

    fig.suptitle(
        "Hydroecological vegetation trends",
        fontsize=15,
        y=1.02
    )

    fig.tight_layout()

    safe_sat = (
        satellite
        .replace(" ", "_")
        .replace("-", "")
    )

    fig_path = (
        outdir
        / f"Hydroecological_trends_{safe_sat}.png"
    )

    fig.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # Export the underlying monthly series too
    steppe_path = (
        outdir
        / f"Steppe_monthly_series_{safe_sat}.csv"
    )

    hydro_path = (
        outdir
        / f"Hydrological_monthly_series_{safe_sat}.csv"
    )

    steppe_monthly.to_csv(
        steppe_path,
        index=False
    )

    hydro_monthly.to_csv(
        hydro_path,
        index=False
    )

    print(
        f"  Saved hydroecological figure: "
        f"{fig_path}"
    )

    return steppe_stats, hydro_stats


# ============================================================
# 10. MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    summary_rows = []

    for satellite in SATELLITES_TO_RUN:

        print("\n" + "=" * 65)
        print(f"PROCESSING {satellite}")
        print("=" * 65)

        df = load_satellite_data(
            satellite
        )

        habitat_monthly = (
            make_monthly_habitat_means(df)
        )

        # ----------------------------------------------------
        # Type 1: Overall
        # ----------------------------------------------------

        overall_stats = plot_overall_trend(
            habitat_monthly=habitat_monthly,
            satellite=satellite,
            outdir=OUTPUT_DIR
        )

        summary_rows.append({
            "Satellite": satellite,
            "Trend_type": "Overall",
            "Group": "All habitats",
            **overall_stats
        })

        # ----------------------------------------------------
        # Type 2: Hydroecological separated
        # ----------------------------------------------------

        steppe_stats, hydro_stats = (
            plot_hydroecological_trends(
                habitat_monthly=habitat_monthly,
                satellite=satellite,
                outdir=OUTPUT_DIR
            )
        )

        summary_rows.append({
            "Satellite": satellite,
            "Trend_type": "Hydroecological",
            "Group": "Steppe habitats",
            **steppe_stats
        })

        summary_rows.append({
            "Satellite": satellite,
            "Trend_type": "Hydroecological",
            "Group": (
                "Spreading area + Wadi and gullies"
            ),
            **hydro_stats
        })

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "Overall_and_hydroecological_trend_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 65)
    print("DONE")
    print(f"Output folder:\n{OUTPUT_DIR}")
    print(f"Summary:\n{summary_path}")
    print("=" * 65)

    return summary_df


# ============================================================
# 11. RUN
# ============================================================

if __name__ == "__main__":
    summary_df = main()
