/** Register the Tool call tree, details renderer, and built-in atomic views. */
import type { ConnectionHandle } from '@deepseek-ai/dsh-client-connection/client'
import type { ClientContext } from '@deepseek-ai/dsh-client-runtime/client'
import type { ClientRemote } from '@deepseek-ai/dsh-api-remotes/client'
import type {} from '@deepseek-ai/dsh-api-remotes/client'
import type {} from '@deepseek-ai/dsh-client-ui-conversation/client'
import { ToolCallTree } from './tool/ToolCallTree.tsx'
import { ToolDetails } from './tool/ToolDetails.tsx'
import { CONVERSATION_NS as NS } from './locale.ts'
import { createKernelAgentLiveLogStore } from './tool/kernelagent-live-log.ts'
import { askQuestionToolview } from './tool/toolviews/ask-question-row.tsx'
import { bashToolviewSample } from './tool/toolviews/bash-sample.tsx'
import { fileMutationToolview } from './tool/toolviews/file-mutation-row.tsx'
import { installKernelAgentToolview } from './tool/toolviews/kernelagent-row.tsx'
import { KernelAgentLogsView } from './tool/toolviews/kernelagent-logs-view.tsx'
import { readToolview } from './tool/toolviews/read-row.tsx'
import { searchToolview } from './tool/toolviews/search-row.tsx'
import { todoToolview } from './tool/toolviews/todo-row.tsx'
import { webToolview } from './tool/toolviews/web-row.tsx'

/** Required services: the slot registry, Host description, and remote event bus. */
export const inject = ['slots', 'connection', 'remote']

/**
 * Mount the whole-Tool renderers and built-in atomic Tool registrations.
 * @param ctx - Client root context.
 */
export function apply(ctx: ClientContext): void {
  const connection = ctx.get('connection') as ConnectionHandle
  const remote = ctx.get('remote') as ClientRemote
  const toolInject = () => ({ hooks: { hostDescription: connection.hostDescription } })
  const kernelAgentLiveLogs = createKernelAgentLiveLogStore()

  // Register the tab before live-log subscription so a remote/$on failure
  // cannot leave the conversation ring without the 日志 entry.
  ctx.slots.inject('conversation.view', () => ctx.slots.register({
    name: 'conversation.view',
    id: 'kernelagent-logs',
    order: 20,
    label: '日志',
    inject: () => ({ hooks: { kernelAgentLiveLogs } }),
  }, KernelAgentLogsView))

  ctx.effect(() => remote.$on('kernelagent/log', (update) => {
    kernelAgentLiveLogs.ingest(update)
  }))

  ctx.slots.inject('conversation.chat.node', () => ctx.slots.register({
    name: 'conversation.chat.node',
    key: 'tool-call',
    locale: NS,
    children: {
      'tool.call.toolview': { kind: 'keyed', scope: 'session' },
    },
    inject: toolInject,
  }, ToolCallTree))

  ctx.slots.inject('conversation.details.tool', () => ctx.slots.register({
    name: 'conversation.details.tool',
    locale: NS,
    inject: toolInject,
  }, ToolDetails))

  ctx.plugin(bashToolviewSample)
  ctx.plugin(readToolview)
  ctx.plugin(fileMutationToolview)
  ctx.plugin(searchToolview)
  ctx.plugin(webToolview)
  ctx.plugin(todoToolview)
  ctx.plugin(askQuestionToolview)
  installKernelAgentToolview(ctx, kernelAgentLiveLogs)
}
