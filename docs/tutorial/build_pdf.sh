#!/usr/bin/env bash
# Regenerate tutorial.pdf from tutorial.md.
# Needs: pandoc + a LaTeX install with xelatex (e.g. `brew install pandoc` and MacTeX/TeX Live).
set -euo pipefail
cd "$(dirname "$0")"

HEADER="$(mktemp -t tutorial_header).tex"
cat > "$HEADER" <<'TEX'
\usepackage{graphicx}
\setkeys{Gin}{width=0.9\linewidth,keepaspectratio}
\usepackage{float}
\floatplacement{figure}{H}   % figures stay where they appear (no floating away from their text)
TEX

pandoc tutorial.md -o tutorial.pdf \
    --pdf-engine=xelatex \
    --resource-path=. \
    -V geometry:margin=1.9cm \
    -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
    -V mainfont="Arial" -V monofont="Menlo" \
    --include-in-header="$HEADER"

rm -f "$HEADER"
echo "wrote tutorial.pdf"
