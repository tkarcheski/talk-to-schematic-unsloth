#!/bin/sh
set -eu

case "${SCHEMATIC_MODEL_KIND:-base}" in
  base) serving_model=/models/base ;;
  adapter)
    /app/.venv/bin/schematic-model deploy bundle \
      --model /models/adapter --base-model /models/base --out /tmp/serving-adapter
    serving_model=/tmp/serving-adapter
    ;;
  *) echo 'SCHEMATIC_MODEL_KIND must be base or adapter' >&2; exit 2 ;;
esac

set -- deploy serve --model "$serving_model" \
  --served-model-name "${SCHEMATIC_MODEL_NAME:-schematic}" \
  --port 8892 --agent-tools \
  --max-image-size "${SCHEMATIC_MAX_IMAGE_SIZE:-2048}" \
  --max-seq "${SCHEMATIC_CONTEXT_LIMIT:-8192}" \
  --max-tokens "${SCHEMATIC_MAX_TOKENS:-1024}" \
  --min-free-vram-gb "${SCHEMATIC_MIN_FREE_VRAM_GB:-10}"
case "${SCHEMATIC_FOUR_BIT:-false}" in
  true) set -- "$@" --4bit ;;
  false) ;;
  *) echo 'SCHEMATIC_FOUR_BIT must be true or false' >&2; exit 2 ;;
esac
exec /app/.venv/bin/schematic-model "$@"
