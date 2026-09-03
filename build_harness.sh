#!/bin/bash
# DeepSeek Harness build script
# Prerequisite: Node.js 24.x and pnpm 11.7.0 are installed, and the repository is cloned

set -e

HARNESS_DIR="${1:-/mnt/zj-data/data/tools/deepseek-harness}"

echo "========================================"
echo " DeepSeek Harness Build"
echo " Directory: ${HARNESS_DIR}"
echo "========================================"

cd "${HARNESS_DIR}"

echo ""
echo "[1/3] Cleaning up old dependencies..."
rm -rf node_modules package-lock.json
npm cache clean --force 2>/dev/null || true

echo ""
echo "[2/3] Installing dependencies (pnpm install)..."
pnpm install

echo ""
echo "[3/3] Building project (pnpm run build)..."
pnpm run build

echo ""
echo "========================================"
echo " Build completed"
echo "========================================"
echo ""
echo "Startup command:"
echo "  cd ${HARNESS_DIR}"
echo "  export DEEPSEEK_API_KEY=\$(python3 -c 'import json; print(json.load(open(\"./scratch-plugin/src/secrets.json\"))[\"deepseekApiKey\"])')"
echo "  pnpm dsh web --patch ./scratch-plugin/src/cordis.yml --port 7890"
echo ""
