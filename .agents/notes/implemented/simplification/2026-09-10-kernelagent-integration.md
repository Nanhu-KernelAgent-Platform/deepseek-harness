# Agent Note: KernelAgent integration ownership

Status: implemented

English | [中文](2026-09-10-kernelagent-integration.zh.md)

## Problem

KernelAgent settings copied into files and environment variables can diverge from saved settings. Source bundles rendered into Markdown and parsed back into files can lose content. An ignored bridge outside the repository prevents reproducing integration fixes from a checkout.

## Decision

Tool execution reads the registered settings namespace and resolves credentials at call time. Source files travel as structured presentation metadata; historical reports retain a replay path. The Harness owns its Python bridge, while KERNELAGENT_WORKING_DIR selects the external KernelAgent project. Startup generates a temporary patch with resolved plugin paths and does not rebuild UI bundles.

## Alternatives considered

**File and environment synchronization.** Multiple copies require ordering and stale-state handling without adding a needed consumer.

**Markdown as the source transport.** Code containing fences can collide with report syntax. Markdown remains only for historical reports and the legacy example path.

## Consequences

Generate returns correctness without new benchmarks; optimize reports existing timings. Synthetic performance charts are omitted. The legacy example fallback remains separate from fresh generation. Focused tests cover configuration precedence, structured source preservation and live expansion; a local startup smoke covers plugin loading. Real model generation is not part of the keyless checks.
