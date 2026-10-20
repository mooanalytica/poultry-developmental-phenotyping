# Feature-Level Multimodal Fusion for Developmental Phenotyping in Laying Hens

**Daniel Edison Essien, Suresh Neethirajan**

Code, results and figures accompanying the manuscript (Revision 4).

## Overview

Hourly video (morphometry, spatial behaviour, optical flow) and audio (bioacoustics,
acoustic level) features from three rooms of one laying-hen flock are fused to estimate
flock age. Models are evaluated with leave-one-room-out and leave-one-age-block-out
validation. All feature normalisation statistics are computed from training data only
within each fold.

## What this repository reproduces

Every number in the manuscript tables and figures. `run_all.sh` reruns all analyses from the
master table into one results sheet (`results/results_sheet.csv`, one row per number with a
result ID); tables and figures are generated from that sheet only.

Changes in this revision:

- slope features use only the observations inside each window (time-bounded slopes);
- a 12 h purge and embargo around every held-out age block, checked by an independent assertion;
- a past-only camera-scale QC filter replaces the centred one;
- the growth-gap detector is calibrated inside each outer fold;
- the shuffled-label control permutes whole room-day label blocks.

ExtraTrees and random forest run single-threaded for bit-reproducibility.

## Repository structure

```
code/
  run_all.py                       runs every analysis and writes the sheet, tables and manifest
  make_tables.py                   manuscript tables from the sheet
  make_figures.py                  manuscript figures from the sheet (python code/make_figures.py)
  common.py, datasets.py, models.py, preprocessing.py, scale_filter.py
                                   shared modules (feature blocks, folds, purge, QC filter)
  detection.py, attribution.py, shuffled_label.py, availability_regimes.py, gru_comparator.py
                                   analyses used by run_all.py
  analysis/, phase2/               analysis and stage scripts called by run_all.py
  tests/                           regression tests
  extract_morphometrics.sh, morpho_hourly.py, build_master.py
                                   how the master table was built from raw sensor outputs
data/                              input data (see Data)
results/                           results sheet (csv, xlsx), tables, fold manifests,
                                   feature dictionary, manifest (hashes and versions), reference/
figures/                           final figures (PNG, PDF, SVG) and captions.md
run_all.sh                         runs the full pipeline
```

## Data

Input data are included in `data/`:

```
data/master_hourly_oct15.csv               fused hourly master table used by the pipeline
data/master_hourly.csv                     original master table
data/audio/audio_hourly_<room>_denoised.csv hourly audio features per room
data/thermal/thermal_frame_features.csv    thermal frame features
```

`master_hourly_oct15.csv` is the master table with the 81 restored hours.
Open data sample: <https://zenodo.org/records/18735885>.

## Usage

```bash
pip install -r requirements.txt
bash run_all.sh
```

The full run takes about 45 to 60 minutes on a laptop. Results are in `results/`
(`results_sheet.csv`, `results_sheet_final.xlsx`, `tables/`). Regenerate the figures with
`python code/make_figures.py`. The regression tests run with `python code/tests/test_regression.py`.

## License

See [LICENSE](LICENSE).
