#!/usr/bin/env bash
# Scrape Minute Cryptic daily puzzles. Resumable. Run from mc_data dir.
set -u
cd "$(dirname "$0")"
COOKIE=$(cat cookie.txt)
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:151.0) Gecko/20100101 Firefox/151.0"
mkdir -p raw_json
LOG=scrape.log
: > "$LOG"
total=$(wc -l < dates_par.txt)
n=0; ok=0; skip=0; fail=0; authfail=0
while read -r date par; do
  n=$((n+1))
  out="raw_json/${date}.json"
  # skip if already have valid json with an answer
  if [ -s "$out" ] && jq -e '.answer' "$out" >/dev/null 2>&1; then
    skip=$((skip+1)); continue
  fi
  # 1) authenticated archive page -> puzzleId
  html=$(curl -s --compressed -b "$COOKIE" -A "$UA" \
      -H "Referer: https://www.minutecryptic.com/archive" \
      "https://www.minutecryptic.com/archive/${date}")
  pid=$(printf '%s' "$html" | grep -oE 'puzzleId\\":\\"[0-9a-f-]{36}' | head -1 | grep -oE '[0-9a-f-]{36}')
  if [ -z "$pid" ]; then
    if printf '%s' "$html" | grep -qi 'subscribe\|Redirecting\|sign-in\|login'; then
      authfail=$((authfail+1))
      echo "AUTHFAIL $date (consecutive=$authfail)" >>"$LOG"
      if [ "$authfail" -ge 5 ]; then
        echo "ABORT: 5 consecutive auth failures - cookie likely expired. Stopped at $date ($n/$total)." >>"$LOG"
        echo "ABORTED_AUTH"
        exit 2
      fi
    else
      fail=$((fail+1)); echo "NOPID $date" >>"$LOG"
    fi
    continue
  fi
  authfail=0
  # 2) open id endpoint -> clean json
  json=$(curl -s --compressed "https://www.minutecryptic.com/api/daily_puzzle/id/${pid}")
  if printf '%s' "$json" | jq -e '.answer' >/dev/null 2>&1; then
    printf '%s' "$json" > "$out"
    ok=$((ok+1))
  else
    fail=$((fail+1)); echo "BADJSON $date $pid" >>"$LOG"
  fi
  if [ $((n % 25)) -eq 0 ]; then
    echo "progress $n/$total ok=$ok skip=$skip fail=$fail" >>"$LOG"
  fi
done < dates_par.txt
echo "DONE total=$total ok=$ok skip=$skip fail=$fail" >>"$LOG"
echo "DONE ok=$ok skip=$skip fail=$fail"
