HARNESS_DIR="${1:-/mnt/zj-data/data/tools/deepseek-harness}"
cd "${HARNESS_DIR}"

# KernelAgent runtime paths (inject into DSH Node process)
export KERNELAGENT_PYTHON=python3
export KERNELAGENT_BRIDGE=/mnt/zj-data/data/zhengty/KernelAgent-from-git/kernelagent_bridge.py
export KERNELAGENT_WORKING_DIR=/mnt/zj-data/data/zhengty/KernelAgent-from-git

export DEEPSEEK_API_KEY=$(python3 -c "import json,sys; print(json.load(open('./scratch-plugin/src/secrets.json'))['deepseekApiKey'])")

pnpm --filter @deepseek-ai/dsh-client-ui-settings-plugins bundle

pnpm dsh web --patch ./scratch-plugin/src/cordis.yml --port 7890
