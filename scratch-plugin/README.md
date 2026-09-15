# KernelAgent integration

English | [中文](README.zh.md)

The settings card is the source of model, backend and generation options. Each tool call reads current settings and resolves credentials before launching Python. No config.json or process-global credential synchronization is used.

Run `pnpm run build` after changing client sources, then `./start_dsh.sh`. The launcher resolves the patch template to a temporary file because the loader resolves relative plugin names against the profile directory. Set `DSH_PORT` to change the listening port.

`KERNELAGENT_WORKING_DIR` selects the KernelAgent checkout (default: the sibling KernelAgent-from-git directory); `KERNELAGENT_PYTHON` selects its Python environment. The versioned bridge is `kernelagent_bridge.py` here; `KERNELAGENT_BRIDGE` optionally selects another bridge. Credentials come from the settings card or its configured credential reference. The optional DeepSeek chat tool uses DEEPSEEK_API_KEY.

Generated files are stored in tool presentation metadata and displayed with per-file downloads. Only the concise status reaches the next model request. A request for backward or gradients is prepared as one bound forward/backward reference, then one generate call creates, verifies, and binds the complete native bundle; automatic optimization then runs separate forward and backward passes with direction-specific latency and MCU measurements. Generate reports correctness without adding benchmarks; optimize reports existing optimization measurements. Historical Markdown reports remain readable. The legacy run_example mode can return cached fallback artifacts and is not evidence of a fresh successful optimization; synthetic charts are not emitted.

Run plugin tests with `pnpm vitest run --config scratch-plugin/vitest.config.mts`.
