# Eastern Morocco Annual Habitat Classification Scripts

This package contains two standalone Python workflows:

- `sentinel2_annual_habitat_classification.py`
- `landsat_annual_habitat_classification.py`

Both scripts use the September-August habitat-year definition, the finalized 11-class legend, four compact spectral indices, Copernicus GLO-30 terrain variables, polygon-grouped five-fold cross-validation, and a 500-tree Earth Engine Smile Random Forest classifier.

## Environment

Activate the analysis environment and run either script from Anaconda Prompt or the JupyterLab terminal:

```bash
conda activate python-gis-2026
python sentinel2_annual_habitat_classification.py
```

or:

```bash
conda activate python-gis-2026
python landsat_annual_habitat_classification.py
```

Required Python packages are `earthengine-api`, `geemap`, `numpy`, `pandas`, and `scikit-learn`.

## Important configuration

Edit only the `USER CONFIGURATION` section near the beginning of each script before running it. The current defaults already use the project assets developed in this workflow:

```text
projects/rse-global-wetlands/assets/Eastern_Morocco_Working_Area
projects/rse-global-wetlands/assets/Eastern_Morocco_RF_Points_HY2025
projects/rse-global-wetlands/assets/Eastern_Morocco_Terrain_GLO30
projects/rse-global-wetlands/assets/Eastern_Morocco_RF_Samples_HY2025
```

The scripts use fixed UTM Zone 30N export grids so all annual products from the same sensor are exactly aligned.

## Sentinel-2 workflow

Default period: HY2018-HY2025 at 10 m.

Predictors:

```text
B2, B3, B4, B8, B11, B12
EVI, MSAVI, NDWI, BSI
Elevation, Slope, Northness, Eastness
```

The script reuses the existing HY2025 sample asset, runs two local CV models, saves accuracy tables, trains and exports the 500-tree GEE classifier, and exports annual maps to Earth Engine Assets and Google Drive.

The two local CV models are:

1. `Features6_Leaf2_MapMatched`: the unweighted configuration corresponding most closely to the operational GEE map.
2. `Balanced_Features6_Leaf2_Sensitivity`: the class-weighted sensitivity model that previously produced the best balanced accuracy and macro F1.

## Landsat workflow

Default period: HY2008-HY2025 at 30 m using Landsat 5, 7, 8, and 9 Collection 2 Tier 1 Level-2 surface reflectance.

Common predictors:

```text
Blue, Green, Red, NIR, SWIR1, SWIR2
EVI, MSAVI, NDWI, BSI
Elevation, Slope, Northness, Eastness
```

The Landsat script creates a sensor-specific HY2025 sample asset if one does not already exist. It retains one deterministic reference record per fixed 30 m grid cell, reducing repeated sampling of the same Landsat pixel. It also exports annual valid-observation-count assets by default, which are important for identifying years or pixels affected by sparse clear observations and Landsat 7 SLC-off gaps.

### Optional cross-sensor adjustment

The Landsat script includes this switch:

```python
APPLY_OLI_TO_ETM_HARMONIZATION = False
```

When set to `True`, Landsat 8/9 reflectance is transformed to an ETM+-like spectral space using the surface-reflectance ordinary-least-squares equations reported by Roy et al. (2016). The resulting assets receive an `ETMlike` suffix, preventing accidental mixing with the default `NativeCommon` series.

This adjustment should be treated as a sensitivity test rather than automatically accepted. The equations were developed for Landsat 7/8 under a specific processing and geographic context, and their transfer to Collection 2 data, Landsat 9, and Eastern Moroccan drylands should be assessed in sensor-overlap years.

## Accuracy outputs

Each script writes local CSV files containing:

- grouped five-fold CV model summaries;
- fold-level OA, kappa, and balanced accuracy;
- pooled class precision, recall, and F1;
- pooled out-of-fold confusion matrices;
- polygon-majority accuracy metrics;
- cross-validation variable importance;
- GEE training and out-of-bag diagnostics;
- GEE training confusion matrix and per-class metrics;
- an annual map manifest;
- a JSON copy of the complete run configuration.

The reportable map accuracy is the polygon-grouped five-fold CV result for `Features6_Leaf2_MapMatched`. The GEE training accuracy and out-of-bag accuracy are internal diagnostics and should not be presented as independent map accuracy.

## Export behavior

By default, both scripts:

1. export the trained classifier to an Earth Engine asset;
2. export each annual classified map to an Earth Engine asset;
3. wait for those asset exports to finish;
4. load the completed assets and launch cloud-optimized GeoTIFF exports to Google Drive.

Google Drive tasks are launched but the script does not wait for all of them unless this is changed:

```python
WAIT_FOR_DRIVE_EXPORTS = True
```

Existing assets are reused. To deliberately replace them, set:

```python
OVERWRITE_ASSETS = True
```

Use this carefully because the script will delete an existing destination asset before recreating it.

## Scientific limitations to retain in the manuscript

The HY2025 grouped validation is not an independent accuracy assessment for every annual map. Earlier classifications represent temporal transfer of a model calibrated from HY2025 reference data. The Landsat record before 2013 also involves cross-sensor transfer from the OLI-era training composite to TM/ETM+ imagery. Selected years should therefore be evaluated against temporally matched very-high-resolution imagery or other independent evidence before annual transitions are treated as confirmed habitat change.

The initial scripts use a static annual median composite. Seasonal and phenological predictors can be added later if the baseline annual maps show insufficient discrimination among steppe physiognomies or ephemeral water/saline habitats.
