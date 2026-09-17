import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'

let apply: (ctx: any) => void

let mergeKernelAgentConfig: (args: any, globalConfig: any) => Record<string, unknown>

let resolveDescribeKind: (kind: unknown, describe1: string) => string
let buildDescribe2UserMessage: (describe1: string, kind?: 'forward' | 'forward_backward') => string
let buildKernelAgentPrompt: (describe1: string, describe2: string, kind?: 'forward' | 'forward_backward') => string
beforeAll(async () => {
  vi.stubEnv('KERNELAGENT_BRIDGE', '/tmp/kernelagent_bridge.py')
  vi.stubEnv('KERNELAGENT_WORKING_DIR', '/tmp')
  ;({ resolveDescribeKind, buildDescribe2UserMessage } = await import('../src/kernelagent-describe-tool.ts'))
  ;({ buildKernelAgentPrompt } = await import('../src/kernelagent-prompt-store.ts'))
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
      auto_optimize: false,
      auto_inject: false,
      inject_deploy_dir: undefined,
      inject_train_script: undefined,
      inject_train_args: undefined,
      inject_op_name: undefined,
      inject_verify: true,
      generation_max_rounds: 8,
      verify: true,
      platform: 'musa',
      kernel_backend: 'musa',
      reasoning_effort: 'medium',
      enable_experience_memory: true,
      strategy: 'greedy',
      baseURL: 'https://zapi.deuo.top',
    })
  })

  it('takes automatic optimization and separate generation budget from saved settings', () => {
    expect(mergeKernelAgentConfig({ auto_optimize: false }, {
      autoOptimize: true, generationMaxRounds: 3, maxRounds: 7,
    })).toMatchObject({ auto_optimize: true, generation_max_rounds: 3, max_rounds: 7 })
  })

  it('takes Workspace injector settings from the saved configuration', () => {
    expect(mergeKernelAgentConfig({}, {
      autoInject: true, injectDeployDir: '.kernelagent/runtime-custom',
      injectTrainScript: 'train.py', injectTrainArgs: '--epochs 1',
      injectOpName: 'relu', injectVerify: false,
    })).toMatchObject({
      auto_inject: true, inject_deploy_dir: '.kernelagent/runtime-custom',
      inject_train_script: 'train.py', inject_train_args: '--epochs 1',
      inject_op_name: 'relu', inject_verify: false,
    })
  })

  it('ignores blank model values and uses a safe fallback', () => {
    expect(mergeKernelAgentConfig({ model: '   ' }, { modelName: '' }).model)
      .toBe('deepseek-chat')
  })
})


describe('combined forward/backward preparation', () => {
  it('infers one bound task when the dialog requests backward generation', () => {
    expect(resolveDescribeKind(undefined, '生成正向并自动推导反向算子')).toBe('forward_backward')
    expect(resolveDescribeKind(undefined, 'generate the forward kernel')).toBe('forward')
  })

  it('instructs KernelAgent to bind once and optimize each direction separately', () => {
    const message = buildDescribe2UserMessage('output = x * x', 'forward_backward')
    expect(message).toContain('Derive the backward formula')
    expect(message).toContain('optimize each direction separately')

    const prompt = buildKernelAgentPrompt(
      'output = x * x',
      'class Model(torch.nn.Module): ...',
      'forward_backward',
    )
    expect(prompt).toContain('one bound operator implementation with forward and derived backward')
  })
})
describe('KernelAgent result projection', () => {
  const definition = () => {
    let tool: any

    apply({ systemPrompt: { section: () => {} }, tools: { register: (value: any) => { tool = value } }, on: () => {} })
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

  it('shows automatic optimization measurements and preserves failure diagnostics', () => {
    const tool = definition()
    const value = { success: true, optimization_status: 'completed', total_rounds: 3, initial_time_ms: 4, best_time_ms: 2 }
    expect(tool.output.render({ mode: 'generate' }, value)[0].text).toContain('2.000x')
    const profiled = { ...value, bottleneck: 'memory', compute_sol_pct: 33.3, memory_sol_pct: 81.3 }
    expect(tool.output.render({ mode: 'generate' }, profiled)[0].text)
      .toContain('MCU utilization: Compute SOL 33.3%, Memory SOL 81.3%')
    expect(tool.output.presentationMeta({ mode: 'generate' }, value).kernelagentChart).toBeDefined()
    const directional = { success: true, optimization_status: 'completed', directional_optimizations: {
      forward: { success: true, pytorch_baseline_ms: 8, initial_time_ms: 4, best_time_ms: 2 },
      backward: { success: true, pytorch_baseline_ms: 12, initial_time_ms: 6, best_time_ms: 3 },
    } }
    expect(tool.output.render({ mode: 'generate' }, directional)[0].text)
      .toContain('Forward PyTorch baseline: 8.000ms; speedup vs PyTorch: 4.000x')
    expect(JSON.parse(tool.output.presentationMeta({ mode: 'generate' }, directional).kernelagentChart).data).toEqual([
      { label: 'Forward PyTorch', value: 8 }, { label: 'Forward initial', value: 4 }, { label: 'Forward best', value: 2 },
      { label: 'Backward PyTorch', value: 12 }, { label: 'Backward initial', value: 6 }, { label: 'Backward best', value: 3 },
    ])
    expect(tool.output.render({ mode: 'generate' }, {
      success: true, optimization_status: 'failed', optimization_error: 'GPU unavailable',
    })[0].text).toContain('verified generated kernel retained')
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
