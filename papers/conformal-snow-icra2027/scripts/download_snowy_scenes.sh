#!/usr/bin/env bash
# Downloads the Snowy Scenes dataset archive (ROADVIEW5k.zip, ~49GB) from
# the authors' OneDrive/SharePoint share link.
#
# Why this isn't a plain `curl <url> -o file.zip`: the share link enforces
# an interactive-login wall for a bare request (curl's default User-Agent
# gets a 401 "Access denied ... you must first browse to the web site and
# select the option to login automatically"). It turns out to be an
# anonymous/tenant-wide link, not one restricted to specific accounts --
# the 401 is bot-detection, not a real auth requirement. A browser-like
# User-Agent on a first GET against the share link is enough to receive an
# anonymous FedAuth session cookie, which a second request (against the
# actual file-download endpoint) can then reuse to get the real bytes.
# Reproduced and verified working end-to-end (curl exit 0, correct
# Content-Length, valid zip central directory) before this script was
# written.
#
# Usage:
#   ./download_snowy_scenes.sh [output_path]
#
# Safe to re-run: uses curl -C - (resume) plus --retry, so an interrupted
# or partial download continues rather than restarting from zero.
set -euo pipefail

OUTPUT_PATH="${1:-ROADVIEW5k.zip}"

# The dataset authors described this share link as temporary ("a permanent
# link is coming soon" as of Sept 2026) -- override via env vars if it
# changes, without needing to edit this script:
SHARE_URL="${SNOWY_SCENES_SHARE_URL:-https://hhse-my.sharepoint.com/:u:/g/personal/abu-mohammed_raisuddin_hh_se/IQCIl8F02Lx-S4yTh_wfqhlLAbKuBvs5dhN4UquMNxuBoGc?e=C4Wwwq}"
DOWNLOAD_SOURCE_URL="${SNOWY_SCENES_SOURCE_URL:-%2Fpersonal%2Fabu%2Dmohammed%5Fraisuddin%5Fhh%5Fse%2FDocuments%2FROADVIEW5k%2Ezip}"
DOWNLOAD_HOST="${SNOWY_SCENES_DOWNLOAD_HOST:-https://hhse-my.sharepoint.com/personal/abu-mohammed_raisuddin_hh_se/_layouts/15/download.aspx}"

UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
WORKDIR="$(mktemp -d)"
COOKIEJAR="$WORKDIR/cookies.txt"
trap 'rm -rf "$WORKDIR"' EXIT

echo "==> Warming up anonymous session..."
curl -sSL -c "$COOKIEJAR" -b "$COOKIEJAR" -A "$UA" -H "Accept: text/html" \
  -o "$WORKDIR/anon_page.html" \
  "$SHARE_URL"

echo "==> Downloading to $OUTPUT_PATH (resumable; safe to re-run if interrupted)..."
# Deliberately NOT using curl's own --retry here: against this host, curl's
# --continue-at(-C -) resume offset is computed once when the process starts
# and is NOT recomputed before each internal --retry attempt. Observed live:
# two responses ~90 minutes apart, after the output file had grown from
# 477MB to 14GB in between, both showed the identical
# `Content-Range: bytes 500314112-...` -- i.e. every internal retry re-asked
# for the offset the *process* started at, silently overwriting/discarding
# everything downloaded since, and the file size oscillated instead of
# growing monotonically. A fresh curl process re-stats the output file on
# every launch, so the fix is to retry at the process level (this loop)
# with a single attempt per curl invocation (--retry 0), not inside curl.
#
# --speed-limit/--speed-time: exit 28 if the transfer drops below 20KB/s for
# 60s straight, so a connection that's open but stalled (observed in
# practice) doesn't hang the current attempt indefinitely.
MAX_ATTEMPTS=500
attempt=0
while true; do
  attempt=$((attempt + 1))
  BEFORE_SIZE="$( { wc -c < "$OUTPUT_PATH"; } 2>/dev/null || echo 0)"
  # set -e would otherwise kill the whole script the instant curl exits
  # non-zero, before this loop's own retry handling ever runs -- disable it
  # for just this one command.
  set +e
  curl -SL -C - --retry 0 \
    --speed-limit 20480 --speed-time 60 \
    -c "$COOKIEJAR" -b "$COOKIEJAR" -A "$UA" -H "Accept: */*" \
    -o "$OUTPUT_PATH" -D "$WORKDIR/headers.txt" \
    "${DOWNLOAD_HOST}?SourceUrl=${DOWNLOAD_SOURCE_URL}"
  status=$?
  set -e
  if [ "$status" -eq 0 ]; then
    break
  fi
  AFTER_SIZE="$( { wc -c < "$OUTPUT_PATH"; } 2>/dev/null || echo 0)"
  if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
    echo "==> curl attempt $attempt failed (exit $status) after ${BEFORE_SIZE} -> ${AFTER_SIZE} bytes; giving up after $MAX_ATTEMPTS attempts." >&2
    exit "$status"
  fi
  echo "==> curl attempt $attempt failed (exit $status), ${BEFORE_SIZE} -> ${AFTER_SIZE} bytes; retrying with a fresh process in 5s..." >&2
  sleep 5
done

# The final request is very likely a resumed (206 Partial Content) request,
# whose Content-Length is the bytes remaining from the resume point, not the
# full file size -- comparing that against the total downloaded size always
# looks like a "mismatch" once a download has resumed even once. The true
# total lives in Content-Range's ".../<total>" suffix on a 206 response;
# only fall back to Content-Length for a plain (non-resumed) 200 response.
EXPECTED_SIZE="$(grep -i '^content-range:' "$WORKDIR/headers.txt" | tail -1 | tr -d '\r' | sed -E 's#.*/([0-9]+)$#\1#')"
if [ -z "$EXPECTED_SIZE" ]; then
  EXPECTED_SIZE="$(grep -i '^content-length:' "$WORKDIR/headers.txt" | tail -1 | tr -d '\r' | awk '{print $2}' || true)"
fi
ACTUAL_SIZE="$(wc -c < "$OUTPUT_PATH" | tr -d ' ')"

echo "==> Downloaded $ACTUAL_SIZE bytes (server reported $EXPECTED_SIZE)."
if [ -n "$EXPECTED_SIZE" ] && [ "$ACTUAL_SIZE" != "$EXPECTED_SIZE" ]; then
  echo "WARNING: size mismatch -- the transfer may have been cut short. Re-run this script to resume." >&2
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  echo "==> Verifying zip integrity (reading the central directory, not a full CRC pass)..."
  python3 -c "
import sys, zipfile
try:
    zf = zipfile.ZipFile('$OUTPUT_PATH')
    names = zf.namelist()
    print(f'OK: {len(names)} entries, archive opens cleanly.')
except Exception as e:
    print(f'CORRUPT ARCHIVE: {type(e).__name__}: {e}', file=sys.stderr)
    print('Re-run this script to resume the download.', file=sys.stderr)
    sys.exit(1)
"
fi

echo "==> Done: $OUTPUT_PATH"
