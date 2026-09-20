// @vitest-environment jsdom

import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ToolResultNode } from '@deepseek-ai/dsh-client-runtime/client'
import { makeTranslate } from '@deepseek-ai/dsh-client-test-runtime'
import { zh as commonZh } from '@deepseek-ai/dsh-client-locale/src/locales/zh.ts'
import { zh } from '@deepseek-ai/dsh-client-ui-conversation/src/client/locales.ts'
import { createKernelAgentLiveLogStore } from '../src/client/tool/kernelagent-live-log.ts'
import { KernelAgentRow } from '../src/client/tool/toolviews/kernelagent-row.tsx'
import { KernelAgentLogsView } from '../src/client/tool/toolviews/kernelagent-logs-view.tsx'

afterEach(cleanup)

const t = makeTranslate(zh, commonZh)

describe('KernelAgentRow live completion', () => {
  it('auto-expands when a mounted running call settles successfully', () => {
    const running = {
      callId: 'ka-live', name: 'kernelagent', argsRaw: '{"mode":"generate"}',
      time: 1_000, subCalls: [],
    }
    const settled: ToolResultNode = {
      kind: 'tool-result', seq: 2, time: 2_000, callId: 'ka-live',
      call: { name: 'kernelagent', argsRaw: '{"mode":"generate"}' },
      callTime: 1_000,
      content: [{ type: 'text', text: 'KernelAgent ✅ succeeded' }],
      meta: { kernelagentFiles: [{ fileName: 'kernel.py', source: 'print("settled")' }] },
      isError: false, callView: null, resultView: null, subCalls: [],
    }
    const props = (block: unknown) => ({
      callId: 'ka-live', toolName: 'kernelagent', block, openFile: vi.fn(), t,
    }) as unknown as Parameters<typeof KernelAgentRow>[0]
    const view = render(<KernelAgentRow {...props(running)} />)

    expect(view.queryByText('print("settled")')).toBeNull()
    view.rerender(<KernelAgentRow {...props(settled)} />)
    expect(view.getByText('print("settled")')).toBeTruthy()
  })

  it('shows live log text for the matching callId while running and after settle', () => {
    const store = createKernelAgentLiveLogStore()
    store.ingest({ sessionId: 's1', callId: 'ka-log', text: 'round 1 starting' })
    const useLogs = <T,>(select: (logs: ReturnType<typeof store.getSnapshot>) => T) =>
      select(store.getSnapshot())
    const running = {
      callId: 'ka-log', name: 'kernelagent', argsRaw: '{"mode":"generate"}',
      time: 1_000, subCalls: [],
    }
    const settled: ToolResultNode = {
      kind: 'tool-result', seq: 2, time: 2_000, callId: 'ka-log',
      call: { name: 'kernelagent', argsRaw: '{"mode":"generate"}' },
      callTime: 1_000,
      content: [{ type: 'text', text: 'KernelAgent ✅ succeeded' }],
      meta: {},
      isError: false, callView: null, resultView: null, subCalls: [],
    }
    const props = (block: unknown) => ({
      callId: 'ka-log', toolName: 'kernelagent', block, openFile: vi.fn(), t,
      useKernelAgentLiveLogs: useLogs,
    }) as unknown as Parameters<typeof KernelAgentRow>[0]

    const view = render(<KernelAgentRow {...props(running)} />)
    expect(view.getByText('round 1 starting')).toBeTruthy()
    expect(view.getByText('运行日志')).toBeTruthy()

    store.ingest({ sessionId: 's1', callId: 'ka-log', text: 'round 1 starting\ndone' })
    view.rerender(<KernelAgentRow {...props(settled)} />)
    expect(view.getByText(/done/)).toBeTruthy()
    expect(view.getByText('执行日志')).toBeTruthy()
  })
})

describe('createKernelAgentLiveLogStore', () => {
  it('tracks latest call per session and replaces text per callId', () => {
    const store = createKernelAgentLiveLogStore()
    const seen: string[] = []
    store.subscribe(() => {
      seen.push(store.getSnapshot().byCallId['c1']?.text ?? '')
    })
    store.ingest({ sessionId: 's', callId: 'c1', text: 'a' })
    store.ingest({ sessionId: 's', callId: 'c1', text: 'a' })
    store.ingest({ sessionId: 's', callId: 'c1', text: 'ab' })
    store.ingest({ sessionId: 's', callId: 'c2', text: 'other' })
    expect(seen).toEqual(['a', 'ab', 'ab'])
    expect(store.getSnapshot().latestBySession).toEqual({ s: 'c2' })
    expect(store.getSnapshot().byCallId['c2']?.text).toBe('other')
  })
})

describe('KernelAgentLogsView', () => {
  it('renders the latest session log and an empty hint otherwise', () => {
    const store = createKernelAgentLiveLogStore()
    const useLogs = <T,>(select: (logs: ReturnType<typeof store.getSnapshot>) => T) =>
      select(store.getSnapshot())
    const base = {
      sessionId: 's1',
      useSession: vi.fn(),
      useSessions: vi.fn(),
      useWorkspaces: vi.fn(),
      useKernelAgentLiveLogs: useLogs,
    } as unknown as Parameters<typeof KernelAgentLogsView>[0]

    const empty = render(<KernelAgentLogsView {...base} />)
    expect(empty.getByText('运行 kernelagent 后，实时输出会显示在这里。')).toBeTruthy()

    store.ingest({ sessionId: 's1', callId: 'ka-1', text: 'live pane output' })
    empty.rerender(<KernelAgentLogsView {...base} />)
    expect(empty.getByText('live pane output')).toBeTruthy()
    expect(empty.getByText(/call ka-1/)).toBeTruthy()
  })
})
