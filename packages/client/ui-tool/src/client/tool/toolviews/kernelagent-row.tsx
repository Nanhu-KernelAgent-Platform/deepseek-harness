/** KernelAgent tool row: live bridge logs plus generated source artifacts. */

import { IconCodeOutline16 } from '@deepseek-ai/dsh-client-ui-primitives'
import type { Context } from '@deepseek-ai/cordis'
import type { PropsLocale } from '@deepseek-ai/dsh-client-ui-slots'
import type { ToolCallViewProps } from '../../contract/slots.ts'
import { CONVERSATION_NS as NS } from '../../locale.ts'
import { ToolRow } from '../components/ToolRow.tsx'
import { toolRowModel } from '../models/tool-call-model.ts'
import type { KernelAgentLiveLogMap, KernelAgentLiveLogStore } from '../kernelagent-live-log.ts'
import { KernelAgentArtifacts, type KernelArtifact } from './kernelagent-artifacts.tsx'
import css from './kernelagent-artifacts.module.css'

type KernelAgentRowProps = ToolCallViewProps & PropsLocale<'conversation'> & {
  useKernelAgentLiveLogs?: <T>(select: (logs: KernelAgentLiveLogMap) => T) => T
}

/** Read the replay-safe report projected by kernelagent-tool. */
function reportOf(meta: unknown): string | null {
  if (typeof meta !== 'object' || meta === null) return null
  const report = (meta as { kernelagentReport?: unknown }).kernelagentReport
  return typeof report === 'string' && report.length > 0 ? report : null
}

/** A compact call row whose expanded body owns live logs, source previews, and downloads. */
export function KernelAgentRow({
  callId, toolName, block, inspect, t,
  useKernelAgentLiveLogs = (select) => select({
    byCallId: {},
    latestBySession: {},
  }),
}: KernelAgentRowProps) {
  const model = toolRowModel(toolName, block)
  const liveLog = useKernelAgentLiveLogs(logs => logs.byCallId[callId]?.text ?? '')
  const report = 'kind' in block ? reportOf(block.meta) : null
  const meta = 'kind' in block ? block.meta : null
  const rawFiles = meta && typeof meta === 'object' ? (meta as { kernelagentFiles?: unknown }).kernelagentFiles : null
  const files = Array.isArray(rawFiles) ? rawFiles.filter((file: unknown): file is KernelArtifact =>
    file !== null && typeof file === 'object' && 'fileName' in file && typeof file.fileName === 'string'
    && 'source' in file && typeof file.source === 'string') : []
  const rawChart = meta && typeof meta === 'object' ? (meta as { kernelagentChart?: unknown }).kernelagentChart : null
  const chart = typeof rawChart === 'string' ? rawChart : ''
  const hasArtifacts = files.length > 0 || report !== null || chart !== ''
  const hasLiveLog = liveLog.length > 0
  const status = model.output
    ?.split('\n')
    .find(line => line.startsWith('KernelAgent '))
    ?.replace(/^KernelAgent\s+/, '')
  const logNode = !hasLiveLog
    ? null
    : (
      <div className={css.liveLog}>
        <div className={css.liveLogLabel}>{model.state === 'running' ? '运行日志' : '执行日志'}</div>
        <pre className={css.liveLogBody}>{liveLog}</pre>
      </div>
    )
  const artifactNode = !hasArtifacts
    ? null
    : <KernelAgentArtifacts report={report ?? ''} chart={chart} {...files.length > 0 ? { files } : {}} t={t} />
  const outputNode = logNode === null && artifactNode === null
    ? undefined
    : (
      <div className={css.fileList}>
        {logNode}
        {artifactNode}
      </div>
    )
  return (
    <ToolRow
      t={t}
      variant="code"
      toolName={toolName}
      icon={<IconCodeOutline16 />}
      title="KernelAgent"
      summary={hasLiveLog && model.state === 'running' ? '运行中…' : (status ?? model.summary)}
      body={null}
      output={null}
      defaultExpanded={(hasArtifacts && model.state === 'ok') || hasLiveLog}
      outputNode={outputNode}
      errorSummary={model.errorSummary}
      state={model.state}
      inspect={inspect}
    />
  )
}

/**
 * Register the KernelAgent tool card against a shared live-log store.
 * @param liveLogs - process-local store also fed to the conversation view tab.
 */
export function installKernelAgentToolview(ctx: Context, liveLogs: KernelAgentLiveLogStore): void {
  ctx.slots.inject('tool.call.toolview', () => ctx.slots.register({
    name: 'tool.call.toolview', key: 'kernelagent', locale: NS,
    inject: () => ({ hooks: { kernelAgentLiveLogs: liveLogs } }),
  }, KernelAgentRow))
}
