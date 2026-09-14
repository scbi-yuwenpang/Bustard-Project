# Eastern Morocco annual habitat-classification scripts — v2

This revision fixes the Earth Engine error:

```text
EEException: Computed value is too large
```

which occurred when a 500-tree in-memory classifier was evaluated with:

```python
classifier.confusionMatrix().array().getInfo()
```

## What changed

1. The 500-tree classifier is exported to an Earth Engine classifier asset **before** diagnostics are calculated.
2. The stored classifier is applied to the stored training-sample table through an Earth Engine **batch table export**.
3. Only `class_id`, `prediction`, `feature_id`, and `cv_fold` are saved in the diagnostic asset.
4. The training confusion matrix and related metrics are calculated locally from that small exported prediction table.
5. `classifier.explain()` is reduced server-side to only the OOB estimate and variable importance; the full tree representation is not downloaded.
6. GEE diagnostic failure is non-fatal, so it cannot prevent annual habitat-map exports. The grouped five-fold CV remains the primary accuracy assessment.

## Run

```bash
conda activate python-gis-2026
python sentinel2_annual_habitat_classification_v2.py
```

For Landsat:

```bash
python landsat_annual_habitat_classification_v2.py
```

## New diagnostic assets

Sentinel-2:

```text
projects/rse-global-wetlands/assets/Eastern_Morocco_S2_RF_TrainingPredictions_HY2025_T500
```

Landsat:

```text
projects/rse-global-wetlands/assets/Eastern_Morocco_Landsat_RF_TrainingPredictions_HY2025_T500_<sensor-space>
```

The script reuses existing classifier and diagnostic assets unless `OVERWRITE_ASSETS = True`.
