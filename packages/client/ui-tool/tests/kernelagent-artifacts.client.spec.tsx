// @vitest-environment jsdom

import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { makeTranslate } from '@deepseek-ai/dsh-client-test-runtime'
import { zh as commonZh } from '@deepseek-ai/dsh-client-locale/src/locales/zh.ts'
import { zh } from '@deepseek-ai/dsh-client-ui-conversation/src/client/locales.ts'
import {
  KernelAgentArtifacts, parseKernelArtifacts,
} from '../src/client/tool/toolviews/kernelagent-artifacts.tsx'

afterEach(cleanup)

const t = makeTranslate(zh, commonZh)
const report = [
  '━━━ kernel.py ━━━\n\n```python:kernel.py\nprint("python")\n```',
  '━━━ kernel.mu ━━━\n\n```cpp:kernel.mu\nvoid kernel() {}\n```',
].join('\n\n')

describe('KernelAgentArtifacts', () => {
  it('preserves embedded Markdown fences in structured source files', () => {
    const source = 'doc = """\n```yaml\nFILE: kernel.py\n```\n"""'
    const view = render(<KernelAgentArtifacts files={[{ fileName: 'kernel.py', source }]} t={t} />)
    expect(view.container.querySelector('code')?.textContent).toBe(source)
    expect(view.getByRole('button', { name: '下载' })).toBeTruthy()
  })

  it('parses and renders each generated source as a separate downloadable file', () => {
    expect(parseKernelArtifacts(report).map(file => file.fileName))
      .toEqual(['kernel.py', 'kernel.mu'])

    const view = render(<KernelAgentArtifacts report={report} t={t} />)

    expect(view.getByText('kernel.py')).toBeTruthy()
    expect(view.getByText('kernel.mu')).toBeTruthy()
    expect(view.getByText('print("python")')).toBeTruthy()
    expect(view.getByText('void kernel() {}')).toBeTruthy()
    expect(view.getAllByRole('button', { name: '下载' })).toHaveLength(2)
  })
})
