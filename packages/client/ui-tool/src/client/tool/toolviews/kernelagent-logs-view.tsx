/** Conversation view tab: live KernelAgent bridge output for the current session. */

import { useEffect, useRef } from 'react'
import type { ConvViewProps } from '@deepseek-ai/dsh-client-ui-conversation/client'
import type { KernelAgentLiveLogMap } from '../kernelagent-live-log.ts'
import css from './kernelagent-logs-view.module.css'

export type KernelAgentLogsViewProps = ConvViewProps & {
  useKernelAgentLiveLogs: <T>(select: (logs: KernelAgentLiveLogMap) => T) => T
}

/**
 * Full-pane live log for the session's latest KernelAgent call.
 * @param props - session kit plus the shared live-log hook.
 */
export function KernelAgentLogsView({
  sessionId,
  useKernelAgentLiveLogs,
}: KernelAgentLogsViewProps) {
  const entry = useKernelAgentLiveLogs((logs) => {
    const callId = logs.latestBySession[String(sessionId)]
    return callId === undefined ? undefined : logs.byCallId[callId]
  })
  const bodyRef = useRef<HTMLPreElement>(null)

  useEffect(() => {
    const el = bodyRef.current
    if (el === null) return
    el.scrollTop = el.scrollHeight
  }, [entry?.text])

  if (entry === undefined || entry.text.length === 0) {
    return (
      <div className={css.empty} data-kernelagent-logs="">
        <p className={css.emptyTitle}>KernelAgent 日志</p>
        <p className={css.emptyHint}>运行 kernelagent 后，实时输出会显示在这里。</p>
      </div>
    )
  }

  return (
    <div className={css.root} data-kernelagent-logs="">
      <header className={css.header}>
        <span className={css.title}>KernelAgent 日志</span>
        <span className={css.meta}>call {entry.callId}</span>
      </header>
      <pre ref={bodyRef} className={css.body}>{entry.text}</pre>
    </div>
  )
}
