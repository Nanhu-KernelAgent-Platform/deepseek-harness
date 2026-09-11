# Agent Note: KernelAgent 自动优化

Status: implemented

[English](2026-09-10-kernelagent-auto-optimization.md) | 中文

## Problem

KernelAgent 生成流程在正确性验证后返回，优化设置不会启动性能优化，用户无法通过设置卡片选择完整流程。

## Decision

设置卡片提供默认关闭的自动优化开关，以及独立的生成纠错和性能优化轮数。Python bridge 在一次工具调用内串联两个阶段，将生成测试传给优化器，并将原生多文件源码转换为优化器输入格式。成功时展示优化源码及实测耗时；优化失败时保留已验证的生成结果并明确报告失败状态。

## Alternatives considered

**只通过提示词串联。** 要求模型再次调用工具依赖模型遵从设置，还需要重复传递源码，因此由 bridge 控制执行顺序。

## Consequences

已有设置保持仅生成行为，开启后会增加耗时和优化资源消耗。测试覆盖设置保存、结果展示、失败保留，以及通过真实 bridge 并替换 GPU/LLM 引擎的 Loader 无界面会话快照。这些测试不证明真实 GPU 性能。
