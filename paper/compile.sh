#!/usr/bin/env bash
# Compile the paper. Missing figure PNGs are handled gracefully by \safefig.
set -e
cd "$(dirname "$0")"
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode main.tex
else
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
fi
