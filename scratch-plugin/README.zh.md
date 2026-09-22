# 本地临时工具

[English](README.md) | 中文

本目录现在只保留可选的本地 `deepseek-chat` 开发工具。`resolve-patch.mjs` 会把相对源码路径转换为绝对临时 patch，供 `start_dsh.sh` 使用。

KernelAgent 已作为独立的 `dsh-kernelagent` bundle 放在本仓库同级目录。启动 Harness 前，只需将它安装到 Web profile 一次：

```sh
pnpm dsh plugin --profile web add ../dsh-kernelagent
./start_dsh.sh
```

外置 bundle 负责 Describe、Pipeline、Engine、Injector、Python bridge 及其测试。`KERNELAGENT_WORKING_DIR` 指定 KernelAgent 代码目录，`KERNELAGENT_PYTHON` 指定 Python 环境；默认自动使用 bundle 自带的 bridge。

可选的本地对话工具读取 `DEEPSEEK_API_KEY`。如果环境变量未设置，`start_dsh.sh` 可以从已忽略的 `scratch-plugin/src/secrets.json` 中读取。
