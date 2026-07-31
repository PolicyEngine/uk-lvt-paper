#!/usr/bin/env bash
# Content checks for the built paper, encoding regressions from the July 2026
# pre-send audit. Usage: check_paper.sh <built.pdf> [<committed.pdf>]
# With a second argument, also verifies the committed PDF matches the rebuilt
# one (normalised text diff, ignoring the build-date line).
set -euo pipefail

built="$1"
committed="${2:-}"

fail=0
text=$(pdftotext "$built" -)

# Structural checks
if grep -q "Figure pending" <<<"$text"; then
  echo "FAIL: built PDF contains a 'Figure pending' placeholder (missing figure file)"; fail=1
fi
if grep -q "??" <<<"$text"; then
  echo "FAIL: built PDF contains '??' (unresolved reference)"; fail=1
fi

# Stale-run and misquote regressions (each was shipped once; see audit PR #12)
banned=(
  "323 of 650"
  "every Scottish and Welsh constituency gains"
  "0.69 rather than"
  "37.3 billion"
  "373.2 billion"
  "2.7 per cent of households"
  "fully passed through to rents"
  "statutory poverty thresholds"
)
for pat in "${banned[@]}"; do
  if grep -qi "$pat" <<<"$text"; then
    echo "FAIL: banned phrase present: '$pat'"; fail=1
  fi
done

# Committed-PDF sync check: the PDF in git must match a rebuild of the tex.
if [ -n "$committed" ]; then
  norm() {
    pdftotext "$1" - | python3 -c '
import re, sys
lines = sys.stdin.read().replace("\f", "\n").split("\n")
months = ("January","February","March","April","May","June","July","August","September","October","November","December")
lines = [l for l in lines if not re.fullmatch(r"\s*(%s) 20\d\d\s*" % "|".join(months), l)]
text = re.sub(r"\s+", " ", " ".join(lines)).strip()
print(text)
'
  }
  if ! diff <(norm "$built") <(norm "$committed") >/dev/null; then
    echo "FAIL: committed paper/main.pdf does not match a rebuild of the tex source."
    echo "      Rebuild it (cd paper && tectonic main.tex) and commit the result."
    fail=1
  fi
fi

if [ "$fail" -eq 0 ]; then
  echo "OK: paper content checks passed"
fi
exit "$fail"
