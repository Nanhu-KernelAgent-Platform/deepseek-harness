# Local scratch tools

English | [中文](README.zh.md)

This directory now contains only the optional local `deepseek-chat` development tool. `resolve-patch.mjs` converts its source-relative overlay into an absolute temporary patch used by `start_dsh.sh`.

KernelAgent is maintained as the independent `dsh-kernelagent` bundle next to this repository. Install it into the Web profile once before starting Harness:

```sh
pnpm dsh plugin --profile web add ../dsh-kernelagent
./start_dsh.sh
```

The external bundle owns Describe, Pipeline, Engine, Injector, their Python bridges, and their tests. `KERNELAGENT_WORKING_DIR` selects the KernelAgent checkout; `KERNELAGENT_PYTHON` selects its Python environment. The bundled bridges are used automatically.

The optional local chat tool reads `DEEPSEEK_API_KEY`. If the variable is unset, `start_dsh.sh` may load it from the ignored `scratch-plugin/src/secrets.json` file.
