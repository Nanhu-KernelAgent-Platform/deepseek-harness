import { cp } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, it } from 'vitest'
import { runLoaderSmoke, LOADER_SMOKE_TEST_TIMEOUT_MS } from '@deepseek-ai/dsh-loader-smoke'

it('runs generation and optimization in one real tool invocation', async () => {
  const path = (relative: string) => fileURLToPath(new URL(relative, import.meta.url))
  const config = path('./fixtures/kernelagent/cordis.yml')
  const env = {
    KERNELAGENT_BRIDGE: path('../../../scratch-plugin/kernelagent_bridge.py'),
    KERNELAGENT_WORKING_DIR: '',
    KERNELAGENT_PYTHON: 'python3', DSH_TELEMETRY_DISABLED: '1',
    PYTHONDONTWRITEBYTECODE: '1',
  }
  const result = await runLoaderSmoke({
    label: 'KernelAgent automatic optimization', tempDirPrefix: 'kernelagent-workflow-',
    binScript: path('./fixtures/headless-driver.ts'),
    configPath: config, binArgs: [config, 'Generate and optimize a kernel.'],
    tsconfigPath: path('../../../tsconfig.json'),
    env,
    prepare: async (cwd) => {
      env.KERNELAGENT_WORKING_DIR = cwd
      await cp(path('./fixtures/kernelagent/triton_kernel_agent'), join(cwd, 'triton_kernel_agent'), { recursive: true })
    },
  })
  const rows = result.stdout.split('\n').filter(line => line.startsWith('{')).map(line => JSON.parse(line))
  const final = rows.find(row => row.type === 'result')
  expect(final).toBeDefined()
  await expect(final.output).toMatchFileSnapshot('./snapshots/kernelagent-auto-optimize.txt')
}, LOADER_SMOKE_TEST_TIMEOUT_MS)
