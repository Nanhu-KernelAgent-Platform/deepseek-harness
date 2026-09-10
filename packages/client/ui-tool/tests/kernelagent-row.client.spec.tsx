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

describe('KernelAgentRow', () => {
  it('opens successful generated code by default with a download action', () => {
    const block: ToolResultNode = {
      kind: 'tool-result', seq: 2, time: 2_000, callId: 'ka-1',
      call: { name: 'kernelagent', argsRaw: '{"mode":"run_example"}' },
      callTime: 1_000,
      content: [{ type: 'text', text: 'KernelAgent ✅ succeeded\nPerf: 1.000ms → 0.500ms' }],
      meta: { kernelagentReport: '```python:kernel.py\nprint("fast")\n```' },
      isError: false, callView: null, resultView: null, subCalls: [],
    }
    const view = render(<KernelAgentRow {...({
      callId: 'ka-1', toolName: 'kernelagent', block, openFile: vi.fn(), t,
    } as unknown as Parameters<typeof KernelAgentRow>[0])} />)

    expect(view.getByText('KernelAgent')).toBeTruthy()
    expect(view.getByText('✅ succeeded')).toBeTruthy()
    expect(view.getByText('print("fast")')).toBeTruthy()
    expect(view.getByRole('button', { name: '下载' })).toBeTruthy()
  })
})
