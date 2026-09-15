import { cp } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, it } from 'vitest'
import { runLoaderSmoke, LOADER_SMOKE_TEST_TIMEOUT_MS } from '@deepseek-ai/dsh-loader-smoke'

it('runs bound forward/backward generation and optimization in one real tool invocation', async () => {
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
    configPath: config, binArgs: [config, 'Generate, bind, and optimize a forward and backward kernel.'],
    tsconfigPath: path('../../../tsconfig.json'),
    env,
    prepare: async (cwd) => {
      env.KERNELAGENT_WORKING_DIR = cwd
      await cp(path('./fixtures/kernelagent/triton_kernel_agent'), join(cwd, 'triton_kernel_agent'), { recursive: true })
    },
  })
  const rows = result.stdout.split('\n').filter(line => line.startsWith('{')).map(line => JSON.parse(line))
  const calls = rows
    .filter(row => row.type === 'session_event' && row.event?.type === 'tool/call')
    .map(row => row.event.data.name)
  expect(calls).toEqual(['kernelagent_describe', 'kernelagent'])
  const describeCall = rows.find(
    row => row.type === 'session_event' && row.event?.type === 'tool/call' && row.event.data.name === 'kernelagent_describe',
  )
  expect(JSON.parse(describeCall.event.data.arguments)).toMatchObject({ kind: 'forward_backward' })
  const final = rows.find(row => row.type === 'result')
  expect(final).toBeDefined()
  await expect(final.output).toMatchFileSnapshot('./snapshots/kernelagent-auto-optimize.txt')
}, LOADER_SMOKE_TEST_TIMEOUT_MS)
