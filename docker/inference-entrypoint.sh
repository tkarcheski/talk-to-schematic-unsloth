#!/bin/sh
set -eu
# The bundle copies only the adapter to ephemeral storage and changes its base
# reference to the container's read-only base mount. Originals are never edited.
/app/.venv/bin/schematic-model deploy bundle \
  --model /models/adapter --base-model /models/base --out /tmp/serving-adapter
exec /app/.venv/bin/schematic-model deploy serve \
  --model /tmp/serving-adapter --served-model-name schematic \
  --port 8892 --agent-tools --max-image-size 2048 \
  --max-seq 8192 --max-tokens 1024 --min-free-vram-gb 10
