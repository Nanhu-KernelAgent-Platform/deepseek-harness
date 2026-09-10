import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'

let apply: (ctx: any) => void

let mergeKernelAgentConfig: (args: any, globalConfig: any) => Record<string, unknown>

beforeAll(async () => {
  vi.stubEnv('KERNELAGENT_BRIDGE', '/tmp/kernelagent_bridge.py')
  vi.stubEnv('KERNELAGENT_WORKING_DIR', '/tmp')
  ;({ mergeKernelAgentConfig, apply } = await import('../src/kernelagent-tool.ts'))
})

afterAll(() => vi.unstubAllEnvs())

describe('mergeKernelAgentConfig', () => {
  it('keeps saved settings authoritative over stale tool-call defaults', () => {
    const merged = mergeKernelAgentConfig(
      {
        model: '',
        workers: 4,
        max_rounds: 20,
        platform: 'cuda',
        kernel_backend: 'triton',
        reasoning_effort: 'xhigh',
        strategy: 'beam_search',
      },
      {
        modelName: 'gpt-5.6-sol',
        baseURL: 'https://zapi.deuo.top',
        workers: 1,
        maxRounds: 1,
        platform: 'musa',
        kernelBackend: 'musa',
        reasoningEffort: 'medium',
        strategy: 'greedy',
        verify: true,
        enableExperienceMemory: true,
      },
    )

    expect(merged).toEqual({
      model: 'gpt-5.6-sol',
      workers: 1,
      max_rounds: 1,
      verify: true,
      platform: 'musa',
      kernel_backend: 'musa',
      reasoning_effort: 'medium',
      enable_experience_memory: true,
      strategy: 'greedy',
      baseURL: 'https://zapi.deuo.top',
    })
  })

  it('ignores blank model values and uses a safe fallback', () => {
    expect(mergeKernelAgentConfig({ model: '   ' }, { modelName: '' }).model)
      .toBe('deepseek-chat')
  })
})

describe('KernelAgent result projection', () => {
  const definition = () => {
    let tool: any
    apply({ systemPrompt: { section: () => {} }, tools: { register: (value: any) => { tool = value } } })
    return tool
  }

  it('prefers actual files to legacy concatenated code and excludes source from model text', () => {
    const tool = definition()
    const value = { success: true, kernel_code: 'FILE: kernel.py', files: { 'kernel.py': 'print("source")' } }
    expect(tool.output.presentationMeta({ mode: 'generate' }, value)).toEqual({
      kernelagentFiles: [{ fileName: 'kernel.py', source: 'print("source")' }],
    })
    expect(JSON.stringify(tool.output.render({ mode: 'generate' }, value))).not.toContain('print("source")')
  })

  it('only shows existing performance measurements for optimization', () => {
    const tool = definition()
    const value = { success: true, verification_status: 'passed', initial_time_ms: 4, best_time_ms: 2, pytorch_baseline_ms: 6 }
    expect(tool.output.render({ mode: 'generate' }, value)[0].text).not.toContain('Speedup')
    const text = tool.output.render({ mode: 'optimize' }, value)[0].text
    expect(text).toContain('2.000x')
    expect(text).toContain('3.000x')
    const meta = tool.output.presentationMeta({ mode: 'optimize' }, value)
    expect(JSON.parse(meta.kernelagentChart).data).toEqual([
      { label: 'PyTorch', value: 6 }, { label: 'Initial', value: 4 }, { label: 'Best', value: 2 },
    ])
    expect(tool.output.presentationMeta({ mode: 'generate' }, value).kernelagentChart).toBeUndefined()
  })
})
