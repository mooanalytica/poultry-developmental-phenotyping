## Feature-Level Multimodal Fusion for Developmental Phenotyping in Laying Hens

**Daniel Edison Essien, Suresh Neethirajan**

Code, results and figures accompanying the manuscript.

## Overview

Hourly video (morphometry, spatial behaviour, optical flow) and audio (bioacoustics,
acoustic level) features from three rooms of one laying-hen flock are fused to estimate
flock age. Models are evaluated with leave-one-room-out and leave-one-age-block-out
validation. All feature normalisation statistics are computed from training data only
within each fold.

## Repository structure

```
code/
  common.py                        paths and shared helpers
  extract_morphometrics.sh         subsample detections from tracking outputs
  morpho_hourly.py                 hourly morphometric features
  build_master.py                  fused hourly master table
  datasets.py                      feature blocks, windowing, missing-data regimes
  models.py                        model definition, metrics, validation folds
  preprocessing.py                 fold-restricted normalisation and fold construction
  evaluate.py                      leave-one-room-out and leave-one-age-block-out evaluation
  attribution.py                   SHAP attribution
  stage_breakdown.py               error by developmental stage
  uncertainty.py                   confidence intervals for the headline metrics
  detection.py                     growth-gap detection with injected delays
  controls.py                      negative controls and thermal test
  comparators.py                   alternative models and stage classifier
  sensitivity_hyperparameters.py   hyperparameter sensitivity
  sensitivity_data_availability.py sensitivity to the complete-case restriction
  figures.py                       figures
data/                               input data (see Data below)
results/                           output tables (CSV)
figures/                           figures (PNG, 600 dpi)
run_all.sh                         runs the analysis from the master table
```

## Data

Input data are included in `data/`:

```
data/master_hourly.csv                     fused hourly master table
data/audio/audio_hourly_<room>_denoised.csv hourly audio features per room
data/thermal/thermal_frame_features.csv    thermal frame features
```

## Usage

```bash
pip install -r requirements.txt
bash run_all.sh
```

## License

See [LICENSE](LICENSE).
