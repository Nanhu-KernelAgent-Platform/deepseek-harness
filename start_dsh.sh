#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="${1:-${SCRIPT_DIR}}"
cd "${HARNESS_DIR}"

# KernelAgent runtime paths (inject into DSH Node process)
export KERNELAGENT_PYTHON="${KERNELAGENT_PYTHON:-python3}"
export KERNELAGENT_BRIDGE="${KERNELAGENT_BRIDGE:-${HARNESS_DIR}/scratch-plugin/kernelagent_bridge.py}"
export KERNELAGENT_WORKING_DIR="${KERNELAGENT_WORKING_DIR:-${HARNESS_DIR}/../KernelAgent-from-git}"

if [[ ! -f "${KERNELAGENT_BRIDGE}" ]]; then
  echo "KernelAgent bridge not found: ${KERNELAGENT_BRIDGE}" >&2
  exit 1
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" && -f ./scratch-plugin/src/secrets.json ]]; then
  export DEEPSEEK_API_KEY
  DEEPSEEK_API_KEY="$(python3 -c "import json; print(json.load(open('./scratch-plugin/src/secrets.json'))['deepseekApiKey'])")"
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set; the optional deepseek-chat tool will be disabled." >&2
fi

PATCH_FILE="$(mktemp "${TMPDIR:-/tmp}/kernelagent-patch.XXXXXX.yml")"
trap 'rm -f -- "$PATCH_FILE"' EXIT
node "${HARNESS_DIR}/scratch-plugin/resolve-patch.mjs" > "$PATCH_FILE"
pnpm dsh web --patch "$PATCH_FILE" --port "${DSH_PORT:-7890}"
