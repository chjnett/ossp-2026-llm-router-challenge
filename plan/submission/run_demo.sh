#!/bin/sh
# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

# 데모 씬 4~6을 재현하고 촬영용 로그를 build/demo/에 남긴다.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
IMAGE="ghcr.io/chjnett/ossp-router@sha256:970e504f67f02371ce71393818df2855563a701f1793d1e0984902c5d4e5f4fb"
DEMO="$ROOT/build/demo"
INPUT="$ROOT/data/materialized/dev"

mkdir -p "$DEMO/logs" "$DEMO/submissions"

printf 'SCENE 4  Offline Docker routing\n'
printf 'image: %s\n\n' "$IMAGE"
: > "$DEMO/logs/scene4-docker.txt"
for tier in fast balanced premium; do
  output="$DEMO/run-$tier"
  mkdir -p "$output"
  chmod 0777 "$output"
  printf '$ docker run ... --tier %s\n' "$tier" | tee -a "$DEMO/logs/scene4-docker.txt"
  started=$(date +%s)
  if ! docker run --rm \
    --network none --cpus 2 --memory 2g --memory-swap 2g --pids-limit 32 \
    --read-only --tmpfs /tmp:rw,size=256m \
    -v "$INPUT:/challenge/input:ro" \
    -v "$output:/challenge/output" \
    "$IMAGE" \
    --input /challenge/input/inputs.json \
    --tier "$tier" \
    --output /challenge/output/submission.json \
    > "$DEMO/logs/current-tier.txt" 2>&1; then
    cat "$DEMO/logs/current-tier.txt" | tee -a "$DEMO/logs/scene4-docker.txt"
    exit 1
  fi
  cat "$DEMO/logs/current-tier.txt" | tee -a "$DEMO/logs/scene4-docker.txt"
  elapsed=$(( $(date +%s) - started ))
  cp "$output/submission.json" "$DEMO/submissions/$tier.json"
  printf '   created %s.json (%ss)\n\n' "$tier" "$elapsed" \
    | tee -a "$DEMO/logs/scene4-docker.txt"
done

printf '\nSCENE 5  Official self-check\n'
if ! PYTHONPATH="$ROOT/src" python3 -m ossp_router.cli self-check \
  --input "$INPUT/inputs.json" \
  --outcomes "$ROOT/data/dev/outcomes.json" \
  --submissions "$DEMO/submissions" \
  --report "$DEMO/report.json" \
  > "$DEMO/logs/scene5-self-check.txt" 2>&1; then
  cat "$DEMO/logs/scene5-self-check.txt"
  exit 1
fi
cat "$DEMO/logs/scene5-self-check.txt"

python3 "$ROOT/plan/submission/summarize_demo_report.py" "$DEMO/report.json" \
  | tee -a "$DEMO/logs/scene5-self-check.txt"

printf '\nSCENE 6  Release gate\n'
if ! PYTHONPATH="$ROOT/src" python3 "$ROOT/tools/verify_release.py" \
  --evidence "$ROOT/build/measure-t43/evidence.json" \
  > "$DEMO/logs/scene6-release.txt" 2>&1; then
  cat "$DEMO/logs/scene6-release.txt"
  exit 1
fi
cat "$DEMO/logs/scene6-release.txt"

printf '\nDemo evidence is ready: %s\n' "$DEMO"
