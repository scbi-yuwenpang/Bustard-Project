# Eastern Morocco annual habitat classification - reviewed production scripts

Files:

- `sentinel2_annual_habitat_classification_final.py`
- `landsat_annual_habitat_classification_final.py`

## Sentinel-2

Primary annual series: HY2018-HY2025, 10 m.

The script reuses these existing Earth Engine assets:

- `projects/rse-global-wetlands/assets/Eastern_Morocco_Working_Area`
- `projects/rse-global-wetlands/assets/Eastern_Morocco_RF_Points_HY2025`
- `projects/rse-global-wetlands/assets/Eastern_Morocco_Terrain_GLO30`
- `projects/rse-global-wetlands/assets/Eastern_Morocco_S2_Spectral_HY2025`
- `projects/rse-global-wetlands/assets/Eastern_Morocco_PredictorStack_HY2025`

The discarded `Eastern_Morocco_RF_Samples_HY2025_5px` asset is not used.
Instead, the 14 saved HY2025 predictors are sampled only at the fixed 17,175
reference points and materialized as a lightweight table asset. The same table
is cached locally for scikit-learn.

Operational GEE RF:

- 500 trees
- variables per split = 6
- minimum leaf population = 2
- bag fraction = 0.632
- seed = 42

The script reproduces all six local RF tuning configurations and saves the
five-fold point-level and polygon-majority accuracy tables. The map-matched
unweighted `Features6_Leaf2` configuration is the reportable GEE counterpart;
`Balanced_Features6_Leaf2` is retained as a sensitivity model.

## Landsat

Primary annual series: HY2008-HY2025, 30 m.

The script reuses the same study-area, fixed reference-point and terrain assets.
It creates one Landsat-specific HY2025 predictor asset and a Landsat-specific
sample table. Because multiple 5-point samples can occupy the same 30 m pixel,
ambiguous pixels carrying more than one habitat class are excluded and one
sample is retained per remaining 30 m pixel.

The default series merges Landsat 5, 7, 8 and 9 Collection 2 Tier 1 Level-2
surface reflectance into common Blue/Green/Red/NIR/SWIR1/SWIR2 bands. The
optional OLI-to-ETM+ harmonization switch is OFF by default and should only be
enabled after overlap-year sensitivity testing.

## Main corrections relative to the development notebook

- Uses the official Eastern Morocco ROI, not a temporary training-extent buffer.
- Uses the fixed 5-points-per-polygon asset, not the discarded expensive
  raster-wide stratified sample asset.
- Reuses the already-exported Sentinel-2 HY2025 predictor and terrain assets.
- Uses a single final 500-tree GEE classifier; the old 100-tree/4-variable
  development classifier is removed.
- Removes the tuple bug caused by a trailing comma in an asset-ID assignment.
- Removes `len(ee.FeatureCollection)` calls; Earth Engine collections use
  `.size()` only for small QA checks.
- Removes undefined variables such as `gee_balanced_accuracy`, `gee_macro_f1`
  and `oob_error` from the old notebook sequence.
- Never requests the 500-tree training confusion matrix with synchronous
  `.getInfo()`; training predictions are materialized with a batch table export
  and metrics are calculated locally.
- Exports the classifier as an Earth Engine asset and reloads it before annual
  classification, reducing repeated graph complexity.
- Uses a fixed UTM 30N affine grid for all annual output rasters.
- Uses categorical `mode` pyramiding for habitat map assets.
- Uses Cloud-Optimized GeoTIFF output with NoData 255 for Google Drive maps.
- Throttles Drive exports and creates annual map assets sequentially for safer
  long-running Earth Engine execution.
- For Landsat NDWI, uses an explicit expression rather than
  `normalizedDifference`, avoiding automatic masking of negative scaled SR
  values.
- Separates Landsat 5/7 QA masking from Landsat 8/9 so the OLI cirrus bit is not
  incorrectly treated as meaningful for TM/ETM+.

## Accuracy outputs

Each script writes CSV tables plus one Excel workbook under the configured
`Accuracy` directory. Outputs include:

- six-model 5-fold tuning summary
- fold metrics
- per-class precision / producer accuracy / F1
- confusion matrices
- out-of-fold predictions
- polygon-majority accuracy and per-class metrics
- cross-fold variable importance
- GEE training/resubstitution summary and confusion matrix
- GEE OOB diagnostic when available
- GEE variable importance when available
- combined manuscript-oriented accuracy overview

The local sample table is saved separately under `RF_Sampling`.

## Annual map outputs

Sentinel-2 outputs:

- HY2018-HY2025
- nominal 10 m grid
- Earth Engine assets and Google Drive GeoTIFFs

Landsat outputs:

- HY2008-HY2025
- 30 m grid
- Earth Engine assets and Google Drive GeoTIFFs

Only the full Eastern Morocco map is exported annually. Bouarfa and Al Baten
can be clipped from the aligned annual maps later, avoiding duplicate annual
classification tasks.

## Safe first run

Activate the environment:

```bash
conda activate python-gis-2026
```

Run Sentinel-2:

```bash
python sentinel2_annual_habitat_classification_final.py
```

Run Landsat:

```bash
python landsat_annual_habitat_classification_final.py
```

All rebuild/overwrite flags are `False` by default. Existing materialized
assets are reused. Google Drive export tasks continue server-side unless
`WAIT_FOR_DRIVE_AT_END=True` is enabled.
