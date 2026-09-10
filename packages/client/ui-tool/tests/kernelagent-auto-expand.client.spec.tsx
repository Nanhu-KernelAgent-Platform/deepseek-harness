// @vitest-environment jsdom

import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ToolResultNode } from '@deepseek-ai/dsh-client-runtime/client'
import { makeTranslate } from '@deepseek-ai/dsh-client-test-runtime'
import { zh as commonZh } from '@deepseek-ai/dsh-client-locale/src/locales/zh.ts'
import { zh } from '@deepseek-ai/dsh-client-ui-conversation/src/client/locales.ts'
import { KernelAgentRow } from '../src/client/tool/toolviews/kernelagent-row.tsx'

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
})
