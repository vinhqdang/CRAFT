#!/usr/bin/env bash
# Downloads a representative subset of the Canadian Adverse Driving
# Conditions (CADC) dataset's LABELED data (not raw) from the official
# University of Waterloo server (http://wiselab.uwaterloo.ca/cadcd_data/,
# per https://github.com/mpitropov/cadc_devkit/blob/master/download_cadcd.py).
#
# Full CADC labeled data is ~93GB across 70 drives (per the devkit's own
# route-stats CSV); we download a representative subset instead, chosen from
# the devkit's own per-drive metadata (cadc_dataset_route_stats.csv):
#   - ALL 18 "bare road" drives (Road snow cover = None) from 2018_03_06 and
#     2018_03_07 -- the nominal pool, ~22.8GB.
#   - 14 "snow-covered road" drives (Road snow cover = Covered) from
#     2019_02_27, spread across its 57 available drives -- the degraded
#     pool, ~17.8GB.
# Total ~40.5GB across 32 drives, comparable in scope to the Snowy Scenes
# download. This is a deliberate, disclosed scoping decision (see
# papers/conformal-snow-icra2027/plan.md), not an attempt to use the full
# dataset.
#
# Resumability: deliberately does NOT use curl's own --retry. See
# scripts/download_snowy_scenes.sh for why: curl's -C - resume offset is
# computed once at process start and is not recomputed before an internal
# --retry attempt, which silently discards progress on a flaky connection.
# This script retries at the process level instead (a fresh curl process
# per attempt correctly re-stats the file).
#
# Usage:
#   ./download_cadc.sh [output_dir]
set -euo pipefail

OUTPUT_DIR="${1:-cadcd}"
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
BASE_URL="http://wiselab.uwaterloo.ca/cadcd_data"
MAX_ATTEMPTS=200

# date -> space-separated drive numbers (bare-road nominal pool: all of it;
# snow-covered degraded pool: a 14-drive subset spread across the 57 available).
NOMINAL_DATES="2018_03_06 2018_03_07"
declare -A DRIVES
DRIVES[2018_03_06]="0001 0002 0005 0006 0008 0009 0010 0012 0013 0015 0016 0018"
DRIVES[2018_03_07]="0001 0002 0004 0005 0006 0007"
DRIVES[2019_02_27]="0002 0006 0011 0018 0022 0028 0034 0040 0046 0051 0058 0065 0072 0079"

download_one() {
  # $1 = url, $2 = output path
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    local before_size
    before_size="$( { wc -c < "$out"; } 2>/dev/null || echo 0)"
    set +e
    curl -SL -C - --retry 0 --speed-limit 20480 --speed-time 60 \
      -A "$UA" -o "$out" "$url"
    local status=$?
    set -e
    if [ "$status" -eq 0 ]; then
      return 0
    fi
    local after_size
    after_size="$( { wc -c < "$out"; } 2>/dev/null || echo 0)"
    if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
      echo "==> giving up on $url after $MAX_ATTEMPTS attempts (curl exit $status), ${before_size} -> ${after_size} bytes" >&2
      return "$status"
    fi
    echo "==> attempt $attempt failed for $url (curl exit $status), ${before_size} -> ${after_size} bytes; retrying with a fresh process in 5s..." >&2
    sleep 5
  done
}

mkdir -p "$OUTPUT_DIR"

for date in "${!DRIVES[@]}"; do
  date_dir="$OUTPUT_DIR/$date"
  mkdir -p "$date_dir"

  calib_zip="$date_dir/calib.zip"
  if [ ! -f "$date_dir/.calib_extracted" ]; then
    echo "==> [$date] downloading calib.zip"
    download_one "$BASE_URL/$date/calib.zip" "$calib_zip"
    (cd "$date_dir" && unzip -oq calib.zip && rm -f calib.zip)
    touch "$date_dir/.calib_extracted"
  fi

  for drive in ${DRIVES[$date]}; do
    drive_dir="$date_dir/$drive"
    mkdir -p "$drive_dir"

    ann_path="$drive_dir/3d_ann.json"
    if [ ! -s "$ann_path" ]; then
      echo "==> [$date/$drive] downloading 3d_ann.json"
      download_one "$BASE_URL/$date/$drive/3d_ann.json" "$ann_path"
    fi

    if [ ! -f "$drive_dir/.labeled_extracted" ]; then
      echo "==> [$date/$drive] downloading labeled.zip"
      labeled_zip="$drive_dir/labeled.zip"
      download_one "$BASE_URL/$date/$drive/labeled.zip" "$labeled_zip"
      echo "==> [$date/$drive] verifying zip integrity"
      python -c "
import zipfile
zf = zipfile.ZipFile('$labeled_zip')
bad = zf.testzip()
if bad is not None:
    raise SystemExit(f'CORRUPT: {bad}')
print(f'OK: {len(zf.namelist())} entries')
"
      (cd "$drive_dir" && unzip -oq labeled.zip && rm -f labeled.zip)
      touch "$drive_dir/.labeled_extracted"
    fi
  done
done

echo "==> Done. Downloaded $(echo $NOMINAL_DATES | wc -w) nominal date(s) + $(echo ${DRIVES[2019_02_27]} | wc -w) degraded drives to $OUTPUT_DIR"
