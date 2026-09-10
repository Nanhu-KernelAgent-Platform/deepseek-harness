# Agent Note: KernelAgent integration ownership

Status: implemented

[English](2026-09-10-kernelagent-integration.md) | 中文

## Problem

将 KernelAgent 配置复制到文件和环境变量会使其偏离已保存的设置。将源码包转换为 Markdown 后再解析可能丢失内容。位于仓库外且被忽略的桥接脚本使集成修复无法随检出复现。

## Decision

工具执行时读取已注册的设置命名空间并解析凭据。源码文件通过结构化展示元数据传递，历史报告保留回放入口。Harness 管理 Python 桥接脚本，KERNELAGENT_WORKING_DIR 指定外部 KernelAgent 项目。启动时生成包含已解析插件路径的临时补丁，不重新构建 UI 包。

## Alternatives considered

**文件与环境变量同步。** 多份副本需要处理更新顺序和过期状态，却没有增加必要的使用方。

**用 Markdown 传输源码。** 源码中的围栏可能与报告语法冲突。Markdown 仅用于历史报告和旧示例流程。

## Consequences

Generate 返回正确性结果而不新增测速，optimize 报告已有耗时。模拟性能图表不输出。旧示例的缓存回退与新生成流程保持区分。定向测试覆盖配置优先级、结构化源码完整性和实时展开，本地启动检查覆盖插件加载。真实模型生成不属于无密钥检查。
