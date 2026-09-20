/**
 * Cordis event for KernelAgent live execution logs. Hosted here so the
 * allowlist and both compiler faces share one declaration; `scratch-plugin`
 * emits and the Tool UI consumes via `ctx.remote.$on`. Process-local only —
 * never appended to the session log.
 *
 * @module @deepseek-ai/dsh-api-remotes/kernelagent-live
 */

/** One throttled whole-buffer push for a single KernelAgent tool call. */
export interface KernelAgentLogUpdate {
  /** Session that owns the running call. */
  sessionId: string
  /** Tool-call id (`exec.callId`), stable across running and settled UI rows. */
  callId: string
  /** Complete stdout/stderr text observed so far for this call (last-wins). */
  text: string
}

declare module '@deepseek-ai/cordis' {
  interface Events {
    /**
     * KernelAgent bridge output for one live tool call. Forwarded verbatim to
     * Web clients; not durable. Emitters throttle and send the full buffer so
     * a late subscriber converges on the latest text without replaying deltas.
     * @param update - session, call, and complete log text so far.
     * @mode emit
     */
    'kernelagent/log'(update: KernelAgentLogUpdate): void
  }
}

export {}
