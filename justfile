# Masters Thesis — Command Cookbook
# Run `just` for the full list of recipes (including submodules).
#
# Layout:
#   just py::<pkg>::<recipe>   — Python monorepo (warp / sdtw / nsde)
#   just tex::<recipe>         — Thesis & presentation builds

mod py 'coding/python/hypothesis'
mod tex 'tex'

# List all available recipes (including submodules)
default:
    @just --list --list-submodules

# ── Top-level shortcuts (documented in CLAUDE.md) ───────────────────────────

sync: py::sync
test: py::test
lint: py::lint

thesis:       tex::build
thesis-watch: tex::watch
open-thesis:  tex::open

pres:  tex::pres-all
all:   tex::all
clean: tex::clean

# Build hypothesis assets and render its presentation.
# wl name:    hypothesis folder under coding/python/hypothesis/scripts/ (e.g. soft-dtw)
# dataset: comma-separated UCR datasets passed to the script (default: all three)
build-asset name dataset="ECG5000":
    bash coding/python/hypothesis/scripts/{{name}}/build_assets.sh {{dataset}}
    just tex::pres {{name}}-hipotese

# ECG5000 EDA — warping premise for Soft-DTW/classification hypothesis
eda-ecg5000:
    just py::eda-ecg5000

# Live preview any .qmd. Defaults to soft-DTW article.
preview file="articles/soft_dtw_ecg5000_report_ptbr.qmd":
    just py::preview {{file}}
