#!/usr/bin/env bash
# End-to-end smoke: run the whole pipeline on a small multi-season subset.
# A few seasons (not one) so every temporal split — train/val/cal/test — is
# non-empty. Used by `docker compose up app` and by CI.
set -euo pipefail

SMOKE_SEASONS="${SMOKE_SEASONS:-2020:2023}"

echo ">> ingest ${SMOKE_SEASONS}"
python -m presnap.ingest --seasons "${SMOKE_SEASONS}"

echo ">> build features"
python -m presnap.build_features

echo ">> train"
python -m presnap.train

echo ">> evaluate"
python -m presnap.evaluate --no-plots

echo "SMOKE OK"
