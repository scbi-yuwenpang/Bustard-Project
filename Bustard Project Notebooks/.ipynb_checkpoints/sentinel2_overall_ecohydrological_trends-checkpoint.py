"""
Sentinel-2 communication trends:
1) Overall trend across monitored habitats
2) Ecohydrological contrast: steppe vs hydrologically influenced habitats

This is a simplified Sentinel-2-only version of the previous communication
plotting workflow.

"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.nonparametric.smoothers_lowess import lowess
from pymannkendall import seasonal_test as seasonal_mk_test
from pymannkendall import original_test as original_mk_test


# ============================================================
# 1. USER SETTINGS
# ============================================================

BASE_DIR = Path(r"C:\Users\PangY\Python-GIS")

INPUT_FILE = (
    BASE_DIR
    / "results" 
    / "RF_Prediction"
    / "rf_pred_sentinel2_26.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "plots"
    / "Plot_RF_Communication"
    / "Sentinel2"
)

VALUE_COL = "F_alive_Pred"
HABITAT_COL = "Habitat_2_"
DATE_COL = "date"

LOWESS_FRAC = 0.30

# Require enough habitat types to contribute to a monthly mean.
# For the overall trend, at least 5 of the 7 habitat types must be present.
OVERALL_MIN_HABITATS = 5

# For the grouped trends:
# - at least 3 of the 5 steppe habitat types
# - both hydrologically influenced habitat types
STEPPE_MIN_HABITATS = 3
HYDRO_MIN_HABITATS = 2


# ============================================================
# 2. HABITAT DEFINITIONS
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

# Normalize a common alternative spelling if it occurs in the CSV.
HABITAT_RENAME = {
    "Wadi & gullies": "Wadi and gullies",
}


# ============================================================
# 3. TREND STATISTICS
# ============================================================

def calculate_trend(
    df_m,
    value_col="mean_cover",
    frac=0.30
):
    """
    Add LOWESS and calculate trend statistics.

    Seasonal Mann-Kendall is preferred for a complete monthly sequence.
    If months are missing, the function falls back to the original
    Mann-Kendall test.
    """

    df_m = (
        df_m
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    if len(df_m) < 12:
        raise ValueError(
            f"Only {len(df_m)} monthly observations are available."
        )

    # --------------------------------------------------------
    # LOWESS
    # --------------------------------------------------------

    x_days = (
        df_m["date"] - df_m["date"].min()
    ).dt.days.to_numpy()

    df_m["lowess"] = lowess(
        df_m[value_col].to_numpy(),
        x_days,
        frac=frac,
        return_sorted=False
    )

    # Approximate LOWESS endpoint change per year
    time_span_years = (
        x_days[-1] - x_days[0]
    ) / 365.25

    if time_span_years > 0:
        slope = (
            df_m["lowess"].iloc[-1]
            - df_m["lowess"].iloc[0]
        ) / time_span_years
    else:
        slope = np.nan

    # Descriptive LOWESS pseudo-R2
    if (
        df_m[value_col].std() > 0
        and df_m["lowess"].std() > 0
    ):
        r = np.corrcoef(
            df_m[value_col],
            df_m["lowess"]
        )[0, 1]
        pseudo_r2 = r ** 2
    else:
        pseudo_r2 = np.nan

    # --------------------------------------------------------
    # Mann-Kendall
    # --------------------------------------------------------

    periods = df_m["date"].dt.to_period("M")

    expected_periods = pd.period_range(
        periods.min(),
        periods.max(),
        freq="M"
    )

    complete_monthly_sequence = (
        len(periods) == len(expected_periods)
        and set(periods) == set(expected_periods)
    )

    try:
        if complete_monthly_sequence and len(df_m) >= 24:
            mk = seasonal_mk_test(
                df_m[value_col].to_numpy(),
                period=12
            )
            mk_method = "Seasonal MK"
        else:
            mk = original_mk_test(
                df_m[value_col].to_numpy()
            )
            mk_method = "Original MK"

        mk_tau = mk.Tau
        mk_p = mk.p
        mk_trend = mk.trend

    except Exception:
        mk_tau = np.nan
        mk_p = np.nan
        mk_trend = "not available"
        mk_method = "not available"

    return df_m, {
        "Slope_per_year": slope,
        "Slope_percentage_points_per_year": slope * 100,
        "Pseudo_R2": pseudo_r2,
        "MK_method": mk_method,
        "MK_tau": mk_tau,
        "MK_p": mk_p,
        "MK_trend": mk_trend,
        "N_months": len(df_m),
    }


# ============================================================
# 4. PREPARE MONTHLY HABITAT MEANS
# ============================================================

def prepare_monthly_habitat_means(
    df,
    value_col=VALUE_COL
):
    """
    Calculate one monthly mean for each habitat.

    This keeps the later combined trends habitat-balanced rather than
    allowing habitats with more stations/records to dominate.
    """

    df = df.copy()

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce"
    )

    df[VALUE_COL] = pd.to_numeric(
        df[VALUE_COL],
        errors="coerce"
    )

    df[HABITAT_COL] = (
        df[HABITAT_COL]
        .astype(str)
        .str.strip()
        .replace(HABITAT_RENAME)
    )

    df = df.dropna(
        subset=[
            DATE_COL,
            VALUE_COL,
            HABITAT_COL
        ]
    )

    df["month"] = (
        df[DATE_COL]
        .dt.to_period("M")
    )

    habitat_monthly = (
        df
        .groupby(
            ["month", HABITAT_COL],
            as_index=False
        )[value_col]
        .mean()
    )

    habitat_monthly["date"] = (
        habitat_monthly["month"]
        .dt.to_timestamp()
    )

    return habitat_monthly


# ============================================================
# 5. BUILD OVERALL + GROUPED SERIES
# ============================================================

def build_overall_series(
    habitat_monthly,
    value_col=VALUE_COL
):
    """
    Equal-weight monthly average across all available habitat types.
    """

    overall = (
        habitat_monthly
        .groupby(
            ["month", "date"],
            as_index=False
        )
        .agg(
            mean_cover=(value_col, "mean"),
            n_habitats=(HABITAT_COL, "nunique")
        )
    )

    overall = overall[
        overall["n_habitats"] >= OVERALL_MIN_HABITATS
    ].copy()

    return overall


def build_group_series(
    habitat_monthly,
    habitats,
    min_habitats
):
    """
    Equal-weight monthly average across habitats in one ecological group.
    """

    d = habitat_monthly[
        habitat_monthly[HABITAT_COL].isin(habitats)
    ].copy()

    grouped = (
        d
        .groupby(
            ["month", "date"],
            as_index=False
        )
        .agg(
            mean_cover=(VALUE_COL, "mean"),
            n_habitats=(HABITAT_COL, "nunique")
        )
    )

    grouped = grouped[
        grouped["n_habitats"] >= min_habitats
    ].copy()

    return grouped


# ============================================================
# 6. SMALL HELPERS FOR PLOT TEXT
# ============================================================

def p_text(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "< 0.001"
    return f"= {p:.3f}"


def trend_label(stats):
    return (
        f"{stats['Slope_percentage_points_per_year']:+.3f} pp yr⁻¹; "
        f"p {p_text(stats['MK_p'])}"
    )


# ============================================================
# 7. MAIN SENTINEL-2 PLOTTING FUNCTION
# ============================================================

def plot_sentinel2_communication_trends(
    df,
    value_col=VALUE_COL,
    frac=LOWESS_FRAC,
    outdir=OUTPUT_DIR
):
    """
    Produce exactly two Sentinel-2 figures:

    Figure 1
        Overall trend across monitored habitat types.

    Figure 2
        Steppe habitats vs hydrologically influenced habitats
        in one overlaid panel.

    Returns
    -------
    summary_df : pandas.DataFrame
    results : dict
    """

    outdir = Path(outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    habitat_monthly = (
        prepare_monthly_habitat_means(
            df,
            value_col=value_col
        )
    )

    # ========================================================
    # A. OVERALL TREND
    # ========================================================

    overall = build_overall_series(
        habitat_monthly,
        value_col=value_col
    )

    overall, overall_stats = calculate_trend(
        overall,
        value_col="mean_cover",
        frac=frac
    )

    # ========================================================
    # B. HYDROECOLOGICAL GROUPS
    # ========================================================

    steppe = build_group_series(
        habitat_monthly,
        habitats=STEPPE_HABITATS,
        min_habitats=STEPPE_MIN_HABITATS
    )

    steppe, steppe_stats = calculate_trend(
        steppe,
        value_col="mean_cover",
        frac=frac
    )

    hydro = build_group_series(
        habitat_monthly,
        habitats=HYDRO_HABITATS,
        min_habitats=HYDRO_MIN_HABITATS
    )

    hydro, hydro_stats = calculate_trend(
        hydro,
        value_col="mean_cover",
        frac=frac
    )

    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    summary_df = pd.DataFrame([
        {
            "Satellite": "Sentinel-2",
            "Group": "All habitats",
            **overall_stats
        },
        {
            "Satellite": "Sentinel-2",
            "Group": "Steppe habitats",
            **steppe_stats
        },
        {
            "Satellite": "Sentinel-2",
            "Group": "Hydrologically influenced habitats",
            **hydro_stats
        },
    ])

    # ========================================================
    # FIGURE 1: OVERALL
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8.6, 4.6)
    )

    ax.scatter(
        overall["date"],
        overall["mean_cover"] * 100,
        s=22,
        alpha=0.35,
        color="0.55",
        edgecolor="none",
        label="Monthly mean"
    )

    ax.plot(
        overall["date"],
        overall["lowess"] * 100,
        linewidth=2.8,
        color="steelblue",
        label="Smoothed trend"
    )

    ax.set_title(
        "Live vegetation cover across monitored habitats",
        fontsize=15,
        weight="bold"
    )

    ax.text(
        0.02,
        0.94,
        "Sentinel-2",
        transform=ax.transAxes,
        fontsize=10.5,
        va="top"
    )

    ax.text(
        0.02,
        0.84,
        (
            f"Trend: {overall_stats['MK_trend']}\n"
            f"{trend_label(overall_stats)}"
        ),
        transform=ax.transAxes,
        fontsize=9.5,
        va="top"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel(
        "Predicted live vegetation cover (%)"
    )

    ax.grid(
        alpha=0.20,
        linestyle="--"
    )

    ax.legend(
        frameon=False,
        loc="upper right"
    )

    fig.tight_layout()

    overall_fig = (
        outdir
        / "01_Sentinel2_overall_trend.png"
    )

    fig.savefig(
        overall_fig,
        dpi=400,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # FIGURE 2: HYDROECOLOGICAL CONTRAST
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8.8, 4.8)
    )

    # Steppe monthly points
    ax.scatter(
        steppe["date"],
        steppe["mean_cover"] * 100,
        s=17,
        alpha=0.16,
        color="firebrick",
        edgecolor="none"
    )

    # Steppe LOWESS
    ax.plot(
        steppe["date"],
        steppe["lowess"] * 100,
        linewidth=2.8,
        color="firebrick",
        label=(
            "Steppe habitats\n"
            + trend_label(steppe_stats)
        )
    )

    # Hydrological monthly points
    ax.scatter(
        hydro["date"],
        hydro["mean_cover"] * 100,
        s=17,
        alpha=0.16,
        color="steelblue",
        edgecolor="none"
    )

    # Hydrological LOWESS
    ax.plot(
        hydro["date"],
        hydro["lowess"] * 100,
        linewidth=2.8,
        color="steelblue",
        label=(
            "Hydrologically influenced habitats\n"
            + trend_label(hydro_stats)
        )
    )

    ax.set_title(
        "Contrasting vegetation trajectories across hydroecological groups",
        fontsize=15,
        weight="bold"
    )

    ax.text(
        0.02,
        0.95,
        "Sentinel-2",
        transform=ax.transAxes,
        fontsize=10.5,
        va="top"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel(
        "Predicted live vegetation cover (%)"
    )

    ax.grid(
        alpha=0.20,
        linestyle="--"
    )

    ax.legend(
        frameon=False,
        fontsize=9.2,
        loc="upper right"
    )

    fig.tight_layout()

    hydro_fig = (
        outdir
        / "02_Sentinel2_hydroecological_trends.png"
    )

    fig.savefig(
        hydro_fig,
        dpi=400,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # EXPORT TABLES
    # ========================================================

    overall.to_csv(
        outdir / "Sentinel2_overall_monthly_series.csv",
        index=False
    )

    steppe.to_csv(
        outdir / "Sentinel2_steppe_monthly_series.csv",
        index=False
    )

    hydro.to_csv(
        outdir / "Sentinel2_hydrological_monthly_series.csv",
        index=False
    )

    summary_df.to_csv(
        outdir / "Sentinel2_communication_trend_summary.csv",
        index=False
    )

    print("\nSaved Sentinel-2 communication outputs to:")
    print(outdir)
    print("\nOverall figure:")
    print(overall_fig)
    print("\nHydroecological figure:")
    print(hydro_fig)

    return summary_df, {
        "overall": overall,
        "steppe": steppe,
        "hydrological": hydro
    }


# ============================================================
# 8. MAIN FUNCTION
# ============================================================

def main():
    """
    Load Sentinel-2 data and run the full workflow.

    Returns
    -------
    summary_df, results
    """

    print(f"Reading:\n{INPUT_FILE}")

    sentinel2_df = pd.read_csv(
        INPUT_FILE
    )

    summary_df, results = (
        plot_sentinel2_communication_trends(
            sentinel2_df,
            value_col=VALUE_COL,
            frac=LOWESS_FRAC,
            outdir=OUTPUT_DIR
        )
    )

    print("\nTrend summary:")
    print(summary_df)

    return summary_df, results


# ============================================================
# 9. RUN AS SCRIPT
# ============================================================

if __name__ == "__main__":
    summary_df, results = main()
