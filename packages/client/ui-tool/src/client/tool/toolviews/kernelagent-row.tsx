/** KernelAgent tool row: keep full generated sources in presentation metadata. */

import { IconCodeOutline16 } from '@deepseek-ai/dsh-client-ui-primitives'
import type { Context } from '@deepseek-ai/cordis'
import type { PropsLocale } from '@deepseek-ai/dsh-client-ui-slots'
import type { ToolCallViewProps } from '../../contract/slots.ts'
import { CONVERSATION_NS as NS } from '../../locale.ts'
import { ToolRow } from '../components/ToolRow.tsx'
import { toolRowModel } from '../models/tool-call-model.ts'
import { KernelAgentArtifacts, type KernelArtifact } from './kernelagent-artifacts.tsx'

type KernelAgentRowProps = ToolCallViewProps & PropsLocale<'conversation'>

/** Read the replay-safe report projected by kernelagent-tool. */
function reportOf(meta: unknown): string | null {
  if (typeof meta !== 'object' || meta === null) return null
  const report = (meta as { kernelagentReport?: unknown }).kernelagentReport
  return typeof report === 'string' && report.length > 0 ? report : null
}

/** A compact call row whose expanded body owns source previews and downloads. */
export function KernelAgentRow({ toolName, block, inspect, t }: KernelAgentRowProps) {
  const model = toolRowModel(toolName, block)
  const report = 'kind' in block ? reportOf(block.meta) : null
  const meta = 'kind' in block ? block.meta : null
  const rawFiles = meta && typeof meta === 'object' ? (meta as { kernelagentFiles?: unknown }).kernelagentFiles : null
  const files = Array.isArray(rawFiles) ? rawFiles.filter((file: unknown): file is KernelArtifact =>
    file !== null && typeof file === 'object' && 'fileName' in file && typeof file.fileName === 'string'
    && 'source' in file && typeof file.source === 'string') : []
  const rawChart = meta && typeof meta === 'object' ? (meta as { kernelagentChart?: unknown }).kernelagentChart : null
  const chart = typeof rawChart === 'string' ? rawChart : ''
  const hasArtifacts = files.length > 0 || report !== null || chart !== ''
  const status = model.output
    ?.split('\n')
    .find(line => line.startsWith('KernelAgent '))
    ?.replace(/^KernelAgent\s+/, '')
  return (
    <ToolRow
      t={t}
      variant="code"
      toolName={toolName}
      icon={<IconCodeOutline16 />}
      title="KernelAgent"
      summary={status ?? model.summary}
      body={null}
      output={null}
      defaultExpanded={hasArtifacts && model.state === 'ok'}
      outputNode={!hasArtifacts
        ? undefined
        : <KernelAgentArtifacts report={report ?? ''} chart={chart} {...files.length > 0 ? { files } : {}} t={t} />}
      errorSummary={model.errorSummary}
      state={model.state}
      inspect={inspect}
    />
  )
}

/** Register the custom KernelAgent result card in the standard keyed Tool slot. */
export const kernelAgentToolview = {
  name: 'kernelagent-toolview',
  inject: ['slots'],
  apply(ctx: Context): void {
    ctx.slots.inject('tool.call.toolview', () => ctx.slots.register({
      name: 'tool.call.toolview', key: 'kernelagent', locale: NS,
    }, KernelAgentRow))
  },
}
