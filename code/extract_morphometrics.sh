#!/bin/bash
# Stream ~250 GB of per-video track_level CSVs and emit a 1-in-N
# subsample of (unix_ts, bbox_h, bbox_w, confidence) per room-season.
# Parsing every row in pandas is infeasible at this volume; subsampling is
# harmless because these rows only ever feed hourly distributional summaries.
#
# Parallelism: the file list is split into P disjoint chunks, each handled by
# its own awk process writing its OWN output file. (Letting several writers
# append to a single stream interleaves and corrupts records.)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/morpho_raw; mkdir -p "$OUT"
N=${N:-100}; CONF=${CONF:-0.30}; P=${P:-8}

for RS in room2_summer room2_fall room6_summer room6_fall room7_summer room7_fall; do
  DIR="data/raw/tracking/$RS/track_level"
  [ -d "$DIR" ] || { echo "skip $RS"; continue; }
  TMP=$(mktemp -d); echo "[$(date +%T)] $RS ..."
  find "$DIR" -name '*.csv' | sort > "$TMP/all.txt"
  NF_=$(wc -l < "$TMP/all.txt")
  # BSD split lacks -n l/K, so round-robin the list ourselves
  awk -v p="$P" -v d="$TMP" '{print > (d "/chunk." (NR%p))}' "$TMP/all.txt"
  for ch in "$TMP"/chunk.*; do
    ( tr '\n' '\0' < "$ch" | xargs -0 awk -v n="$N" -v c="$CONF" -F, '
        FNR==1 {next}
        FNR%n!=0 {next}
        # video_file (field 1) contains commas in many batches, so the row can
        # have 15, 17 or 18 fields. Only the FIRST field is ever split, so we
        # index from the END: unix_ts=NF, confidence=NF-6, bbox_h=NF-7, bbox_w=NF-8.
        NF < 15 {next}
        $(NF-6)+0 < c {next}
        $(NF)=="" {next}
        {printf "%s,%s,%s,%s\n", $(NF), $(NF-7), $(NF-8), $(NF-6)}
      ' > "$ch.out" ) &
  done
  wait
  cat "$TMP"/chunk.*.out > "$OUT/${RS}.csv"
  echo "[$(date +%T)] $RS -> $(wc -l < "$OUT/${RS}.csv") rows from $NF_ files"
  rm -rf "$TMP"
done
