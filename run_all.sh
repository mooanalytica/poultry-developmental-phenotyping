#!/bin/bash
# Run the analysis pipeline from the master table (data/master_hourly.csv) to results/ and figures/.
# Building the master table from raw sensor outputs is a separate step:
#   code/extract_morphometrics.sh -> code/morpho_hourly.py -> code/build_master.py
set -euo pipefail
cd "$(dirname "$0")"
for step in evaluate attribution stage_breakdown uncertainty detection controls comparators \
            sensitivity_hyperparameters sensitivity_data_availability figures; do
  echo "== $step"
  python3 "code/$step.py"
done
