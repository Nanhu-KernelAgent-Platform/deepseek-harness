# Agent Note: KernelAgent automatic optimization

Status: implemented

English | [中文](2026-09-10-kernelagent-auto-optimization.zh.md)

## Problem

KernelAgent generation returns after correctness verification. The optimization settings do not start performance optimization, so users cannot request the complete workflow through the settings card.

## Decision

The settings card exposes an opt-in automatic optimization switch and separate generation and optimization round budgets. The Python bridge sequences both stages in one tool call, carries generated tests into optimization, and renders native source bundles in the optimizer's input format. Successful results project optimized sources and actual timings; optimization failure retains the verified generated output with an explicit failure status.

## Alternatives considered

**Prompt-only chaining.** Asking the model to issue another tool call makes the saved setting dependent on model compliance and duplicates source transfer. The bridge owns the sequence instead.

## Consequences

Existing settings keep generation-only behavior. Enabled calls take longer and consume optimization resources. Tests cover settings persistence, result projection, fallback behavior, and a Loader-based headless transcript using the real bridge with a substituted GPU/LLM engine. Real GPU performance is not established by these tests.
