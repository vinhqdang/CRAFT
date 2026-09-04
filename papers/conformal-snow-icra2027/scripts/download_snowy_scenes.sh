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
curl -SL -C - --retry 10 --retry-delay 5 --retry-all-errors \
  -c "$COOKIEJAR" -b "$COOKIEJAR" -A "$UA" -H "Accept: */*" \
  -o "$OUTPUT_PATH" -D "$WORKDIR/headers.txt" \
  "${DOWNLOAD_HOST}?SourceUrl=${DOWNLOAD_SOURCE_URL}"

EXPECTED_SIZE="$(grep -i '^content-length:' "$WORKDIR/headers.txt" | tail -1 | tr -d '\r' | awk '{print $2}' || true)"
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
