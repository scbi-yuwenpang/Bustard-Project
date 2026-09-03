
"""
gee_climate_export.py

Download monthly area-level climate summaries from Google Earth Engine
for the Morocco study regions.

Climate datasets
----------------
- TerraClimate
- ERA5-Land Monthly Aggregated

Spatial levels
--------------
Regional:
    - Eastern Morocco
    - Tata

Focal:
    - Al Baten
    - Bouarfa

Main use
--------
from gee_climate_export import run_climate_export

task, climate_areas = run_climate_export(
    start_date="1991-01-01",
    end_date=None,
    out_folder="GEE_Clim_exports"
)

The returned Earth Engine export task is started automatically.
"""

import datetime
import ee


# ============================================================
# Default GEE assets
# ============================================================

DEFAULT_AOI_ASSETS = {
    "Eastern Morocco": {
        "asset": "projects/rse-global-wetlands/assets/Eastern_Morocco_Working_Area",
        "spatial_level": "Regional",
    },
    "Tata": {
        "asset": "projects/rse-global-wetlands/assets/Tata_Morocco_Working_Area",
        "spatial_level": "Regional",
    },
    "Al Baten": {
        "asset": "projects/rse-global-wetlands/assets/Al_Baten_study_area",
        "spatial_level": "Focal",
    },
    "Bouarfa": {
        "asset": "projects/rse-global-wetlands/assets/Bouarfa_study_area",
        "spatial_level": "Focal",
    },
}


# ============================================================
# Earth Engine initialization
# ============================================================

def initialize_gee(project=None):
    """
    Initialize Google Earth Engine.

    Parameters
    ----------
    project : str or None
        Optional Google Cloud project ID.

    Notes
    -----
    If authentication has not yet been completed on the machine,
    run ee.Authenticate() once before using this module.
    """
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()


# ============================================================
# Date helpers
# ============================================================

def resolve_end_date(end_date=None):
    """
    Return the exclusive Earth Engine filterDate end date.

    If end_date is None, tomorrow's date is used so that all
    currently available images up to today are included.
    """
    if end_date is None:
        return (
            datetime.date.today() + datetime.timedelta(days=1)
        ).strftime("%Y-%m-%d")

    return end_date


def make_period_label(start_date, end_date=None):
    """
    Create a concise period label for export filenames.
    """
    start_year = str(start_date)[:4]

    if end_date is None:
        return f"{start_year}_Present"

    end_year = str(end_date)[:4]
    return f"{start_year}_{end_year}"


# ============================================================
# Study areas
# ============================================================

def load_study_areas(aoi_assets=None):
    """
    Load study-area assets as Earth Engine FeatureCollections.

    Parameters
    ----------
    aoi_assets : dict or None
        Optional custom AOI dictionary.

    Returns
    -------
    dict
        AOI dictionary with loaded FeatureCollections.
    """
    if aoi_assets is None:
        aoi_assets = DEFAULT_AOI_ASSETS

    aois = {}

    for area_name, info in aoi_assets.items():
        aois[area_name] = {
            "fc": ee.FeatureCollection(info["asset"]),
            "spatial_level": info["spatial_level"],
        }

    return aois


# ============================================================
# Climate preprocessing
# ============================================================

def process_era5_land(img):
    """
    Standardize ERA5-Land Monthly to:
        temp_C
        precip_mm
    """
    temp_c = (
        img.select("temperature_2m")
        .subtract(273.15)
        .rename("temp_C")
    )

    precip = img.select("total_precipitation_sum")

    # Prevent very small negative precipitation values
    precip = precip.where(precip.lt(0), 0)

    precip_mm = (
        precip
        .multiply(1000)
        .rename("precip_mm")
    )

    return (
        temp_c
        .addBands(precip_mm)
        .copyProperties(img, ["system:time_start"])
    )


def process_terraclimate(img):
    """
    Standardize TerraClimate Monthly to:
        temp_C
        precip_mm

    Mean temperature is approximated as:
        (Tmax + Tmin) / 2
    """
    tmax_c = img.select("tmmx").multiply(0.1)
    tmin_c = img.select("tmmn").multiply(0.1)

    temp_c = (
        tmax_c
        .add(tmin_c)
        .divide(2)
        .rename("temp_C")
    )

    precip_mm = (
        img.select("pr")
        .rename("precip_mm")
    )

    return (
        temp_c
        .addBands(precip_mm)
        .copyProperties(img, ["system:time_start"])
    )


def load_climate_datasets(
    start_date="2000-01-01",
    end_date=None,
    datasets=("TerraClimate", "ERA5-Land"),
):
    """
    Load and preprocess selected monthly climate datasets.

    Parameters
    ----------
    start_date : str
        Inclusive start date, e.g. "1991-01-01".

    end_date : str or None
        Exclusive end date for ee.FilterDate.
        If None, all currently available data are requested.

    datasets : iterable
        Any combination of:
        - "TerraClimate"
        - "ERA5-Land"

    Returns
    -------
    dict
        Dataset name -> {"ic": ImageCollection, "scale": meters}
    """
    resolved_end = resolve_end_date(end_date)

    requested = set(datasets)
    output = {}

    if "TerraClimate" in requested:
        terraclimate = (
            ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
            .filterDate(start_date, resolved_end)
            .map(process_terraclimate)
        )

        output["TerraClimate"] = {
            "ic": terraclimate,
            "scale": 4638,
        }

    if "ERA5-Land" in requested:
        era5_land = (
            ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
            .filterDate(start_date, resolved_end)
            .map(process_era5_land)
        )

        output["ERA5-Land"] = {
            "ic": era5_land,
            "scale": 11132,
        }

    unknown = requested.difference({"TerraClimate", "ERA5-Land"})
    if unknown:
        raise ValueError(
            f"Unsupported dataset(s): {sorted(unknown)}. "
            "Use 'TerraClimate' and/or 'ERA5-Land'."
        )

    return output


# ============================================================
# Dataset availability
# ============================================================

def check_collection(ic, name):
    """
    Print first date, last date, and number of monthly images.
    """
    n = ic.size().getInfo()

    if n == 0:
        print(f"{name}: 0 images")
        return {
            "dataset": name,
            "n_images": 0,
            "first_date": None,
            "last_date": None,
        }

    first_date = (
        ee.Date(ic.aggregate_min("system:time_start"))
        .format("YYYY-MM")
        .getInfo()
    )

    last_date = (
        ee.Date(ic.aggregate_max("system:time_start"))
        .format("YYYY-MM")
        .getInfo()
    )

    print(
        f"{name}: {n} monthly images | "
        f"{first_date} to {last_date}"
    )

    return {
        "dataset": name,
        "n_images": n,
        "first_date": first_date,
        "last_date": last_date,
    }


def check_all_datasets(dataset_config):
    """
    Print availability for all selected climate datasets.
    """
    summary = []

    for dataset_name, info in dataset_config.items():
        summary.append(
            check_collection(info["ic"], dataset_name)
        )

    return summary


# ============================================================
# Area extraction
# ============================================================

def extract_to_area(
    ic,
    area_fc,
    area_name,
    spatial_level,
    dataset,
    scale,
):
    """
    Extract monthly spatial mean climate values for one AOI.
    """
    n = ic.size()
    image_list = ic.toList(n)
    indices = ee.List.sequence(0, n.subtract(1))

    def per_image(i):
        img = ee.Image(image_list.get(i))
        date = ee.Date(img.get("system:time_start"))

        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=area_fc.geometry(),
            scale=scale,
            maxPixels=1e13,
        )

        return ee.Feature(
            None,
            stats,
        ).set({
            "dataset": dataset,
            "spatial_level": spatial_level,
            "region": area_name,
            "date": date.format("YYYY-MM"),
            "year": date.get("year"),
            "month": date.get("month"),
        })

    return ee.FeatureCollection(
        indices.map(per_image)
    )


def extract_all_areas(dataset_config, aois):
    """
    Extract all selected climate datasets across all AOIs.
    """
    climate_areas = ee.FeatureCollection([])

    for dataset_name, dataset_info in dataset_config.items():

        for area_name, area_info in aois.items():

            print(
                f"Preparing: {dataset_name} | "
                f"{area_info['spatial_level']} | "
                f"{area_name}"
            )

            fc = extract_to_area(
                ic=dataset_info["ic"],
                area_fc=area_info["fc"],
                area_name=area_name,
                spatial_level=area_info["spatial_level"],
                dataset=dataset_name,
                scale=dataset_info["scale"],
            )

            climate_areas = climate_areas.merge(fc)

    return climate_areas


# ============================================================
# Google Drive export
# ============================================================

def export_climate_areas(
    climate_areas,
    start_date,
    end_date=None,
    out_folder="GEE_Clim_exports",
    filename_prefix=None,
):
    """
    Export the merged climate table to Google Drive.
    """
    period_label = make_period_label(start_date, end_date)

    if filename_prefix is None:
        filename_prefix = (
            f"All_Climate_Areas_Monthly_{period_label}"
        )

    task = ee.batch.Export.table.toDrive(
        collection=climate_areas,
        description=filename_prefix,
        folder=out_folder,
        fileNamePrefix=filename_prefix,
        fileFormat="CSV",
        selectors=[
            "dataset",
            "spatial_level",
            "region",
            "date",
            "year",
            "month",
            "temp_C",
            "precip_mm",
        ],
    )

    task.start()

    print()
    print("Export started.")
    print(f"Google Drive folder: {out_folder}")
    print(f"Output file prefix:  {filename_prefix}")
    print("Check the Earth Engine Tasks tab.")

    return task


# ============================================================
# Main wrapper
# ============================================================

def run_climate_export(
    start_date="2000-01-01",
    end_date=None,
    out_folder="GEE_Clim_exports",
    datasets=("TerraClimate", "ERA5-Land"),
    aoi_assets=None,
    filename_prefix=None,
    gee_project=None,
    initialize=True,
    check_availability=True,
):
    """
    Complete GEE climate extraction and export workflow.

    Parameters
    ----------
    start_date : str
        Inclusive start date.
        Examples:
            "2000-01-01"
            "1991-01-01"
            "1981-01-01"

    end_date : str or None
        Exclusive end date.
        If None, request all data currently available.

    out_folder : str
        Google Drive export folder.

    datasets : tuple/list
        Selected datasets.
        Default:
            ("TerraClimate", "ERA5-Land")

    aoi_assets : dict or None
        Custom AOI assets. If None, default project AOIs are used.

    filename_prefix : str or None
        Optional custom output filename.

    gee_project : str or None
        Optional Google Cloud project for ee.Initialize().

    initialize : bool
        Initialize Earth Engine inside the function.

    check_availability : bool
        Print actual data availability before extraction.

    Returns
    -------
    task : ee.batch.Task
        Started Earth Engine export task.

    climate_areas : ee.FeatureCollection
        Merged monthly climate table before export.

    availability : list
        Dataset availability summaries.
    """

    if initialize:
        initialize_gee(project=gee_project)

    print("=" * 70)
    print("GEE Climate Export")
    print("=" * 70)
    print(f"Requested start date: {start_date}")
    print(
        "Requested end date:   "
        + (end_date if end_date is not None else "latest available")
    )
    print(f"Datasets:             {', '.join(datasets)}")
    print()

    aois = load_study_areas(aoi_assets)

    dataset_config = load_climate_datasets(
        start_date=start_date,
        end_date=end_date,
        datasets=datasets,
    )

    availability = []

    if check_availability:
        print("Dataset availability:")
        availability = check_all_datasets(dataset_config)
        print()

    climate_areas = extract_all_areas(
        dataset_config=dataset_config,
        aois=aois,
    )

    print()
    print(
        "Total output records:",
        climate_areas.size().getInfo(),
    )

    task = export_climate_areas(
        climate_areas=climate_areas,
        start_date=start_date,
        end_date=end_date,
        out_folder=out_folder,
        filename_prefix=filename_prefix,
    )

    return task, climate_areas, availability


# ============================================================
# Example calls
# ============================================================
#
# 1. Current project-period archive
#
# task, climate_areas, availability = run_climate_export(
#     start_date="2000-01-01"
# )
#
#
# 2. Earlier period for a 1991-2020 climate baseline
#
# task, climate_areas, availability = run_climate_export(
#     start_date="1991-01-01"
# )
#
#
# 3. Fixed historical period only
#
# task, climate_areas, availability = run_climate_export(
#     start_date="1991-01-01",
#     end_date="2021-01-01"
# )
#
#
# 4. TerraClimate only
#
# task, climate_areas, availability = run_climate_export(
#     start_date="1991-01-01",
#     datasets=("TerraClimate",)
# )
