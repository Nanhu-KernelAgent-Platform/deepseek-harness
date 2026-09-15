# KernelAgent 集成

[English](README.md) | 中文

设置卡片是模型、后端和生成选项的配置来源。每次工具调用读取当前设置，并在启动 Python 前解析凭据，不使用 config.json 或进程全局凭据同步。

修改客户端源码后运行 `pnpm run build`，再执行 `./start_dsh.sh`。加载器按 profile 目录解析相对插件名，因此启动脚本会将补丁模板解析到临时文件。通过 DSH_PORT 修改监听端口。

KERNELAGENT_WORKING_DIR 指定 KernelAgent 项目，默认是相邻的 KernelAgent-from-git 目录；KERNELAGENT_PYTHON 指定 Python 环境。本目录中的 kernelagent_bridge.py 纳入版本管理，KERNELAGENT_BRIDGE 可指定其他桥接脚本。凭据来自设置卡片或配置的凭据引用，可选的 DeepSeek 对话工具使用 DEEPSEEK_API_KEY。

生成文件保存在工具展示元数据中，支持逐文件下载；只有简短状态进入下一次模型请求。用户要求反向或梯度时，描述阶段会准备一个绑定的正反向参考实现，随后只调用一次 generate，统一生成、验证并绑定完整原生算子包；自动优化再基于各自的耗时和 MCU 数据分别执行正向、反向优化。Generate 报告正确性且不额外测速，optimize 报告已有优化数据。历史 Markdown 报告仍可读取。旧 run_example 模式可能返回缓存产物，不能据此认定本次优化成功；桥接脚本不输出模拟图表。

通过 `pnpm vitest run --config scratch-plugin/vitest.config.mts` 运行插件测试。
