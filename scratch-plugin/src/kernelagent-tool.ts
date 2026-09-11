import { defineTool } from '@deepseek-ai/dsh-tools'
import type { Context } from '@deepseek-ai/cordis'
import { spawn } from 'node:child_process'
import { writeFileSync, readFileSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { CONFIG_TOOL_SETTINGS_NAMESPACE, resolveApiKey, type Config } from './kernelagent-config-tool.ts'
import { peekPreparedPrompt, requiresPreparedPrompt, takePreparedPrompt } from './kernelagent-prompt-store.ts'

// ========== Runtime configuration via environment variables ==========
// These are injected by start_dsh.sh or the host environment.
const PYTHON = process.env.KERNELAGENT_PYTHON || 'python3'
function requiredEnvironment(name: string): string {
  const value = process.env[name]
  if (!value) throw new Error(`[kernelagent-tool] Missing required environment variable: ${name}`)
  return value
}
const BRIDGE = requiredEnvironment('KERNELAGENT_BRIDGE')
const CWD = requiredEnvironment('KERNELAGENT_WORKING_DIR')

function nonEmptyString(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined
  const normalized = value.trim()
  return normalized === '' ? undefined : normalized
}

/** Settings are authoritative; tool-call fields are compatibility fallbacks. */
export function mergeKernelAgentConfig(args: any, globalConfig: any) {
  return {
    model: nonEmptyString(globalConfig.modelName) ?? nonEmptyString(args.model) ?? 'deepseek-chat',
    workers: globalConfig.workers ?? args.workers ?? 4,
    auto_optimize: globalConfig.autoOptimize ?? false,
    generation_max_rounds: globalConfig.generationMaxRounds ?? 8,
    max_rounds: globalConfig.maxRounds ?? args.max_rounds ?? 8,
    verify: globalConfig.verify ?? args.verify ?? true,
    platform: nonEmptyString(globalConfig.platform) ?? nonEmptyString(args.platform) ?? 'musa',
    kernel_backend: nonEmptyString(globalConfig.kernelBackend) ?? nonEmptyString(args.kernel_backend) ?? 'musa',
    reasoning_effort: nonEmptyString(globalConfig.reasoningEffort) ?? nonEmptyString(args.reasoning_effort) ?? 'high',
    enable_experience_memory: globalConfig.enableExperienceMemory
      ?? globalConfig.enable_experience_memory
      ?? args.enable_experience_memory
      ?? true,
    strategy: nonEmptyString(globalConfig.strategy) ?? nonEmptyString(args.strategy) ?? 'beam_search',
    baseURL: nonEmptyString(globalConfig.baseURL) ?? 'https://api.deepseek.com/v1/chat/completions',
  }
}

export const name = 'kernelagent-tool'
export const inject = ['tools', 'systemPrompt', 'settings']

export function apply(ctx: Context) {
  ctx.systemPrompt.section({
    name: 'tool:kernelagent',
    order: 210,
    text: `Use the kernelagent tool to generate, fuse, optimize GPU Triton kernels, or run a predefined example.
Modes: generate | fuse | optimize | run_example. In generate mode, the saved autoOptimize setting automatically runs performance optimization after correctness verification; do not issue a second optimize call for that workflow. Report optimization_status and optimization_error when returned.
For generate, fuse, or optimize: call kernelagent_describe first; those modes refuse to run without its prepared prompt, and problem_code is taken from that prompt (do not invent one). After a successful forward run, if the user also asked for a reverse/backward operator, call kernelagent_describe again with kind=backward, then call kernelagent again.
Call the tool immediately without an announcement or progress preamble. After a successful call, always provide a short final answer in the user's language (2-4 sentences): summarize the requested operator and target platform, report the actual generation and correctness-verification outcome, and mention that the source files can be downloaded from the KernelAgent card above. Include optimization rounds, performance measurements, or MCU bottleneck and SOL utilization only when returned by the tool. Tool success alone is not evidence that correctness verification passed or performance improved; if verification details are absent, say they were not reported. When no optimization was performed (including automatic optimization), summarize correctness only and do not mention missing benchmarks or performance. After explicit or automatic optimization, include the returned optimization timings and speedup, clearly distinguishing the initial-kernel baseline from the PyTorch baseline. Do not run additional benchmarks for the summary. On failure, briefly explain the returned error and any actionable next step supported by it.
KernelAgent settings are authoritative. Do not override model, workers, rounds, platform, backend, strategy, reasoning effort, verification, or experience memory when settings are available.
The KernelAgent tool card displays available generated source files and per-file download buttons. Do not reproduce the report or source code in your reply. The card opens itself; do not tell the user to expand it. The final answer must contain a useful result summary, not just internal reasoning or a statement that the tool succeeded.`,
  })

  // Fail closed when generate/fuse/optimize run without a describe-prepared prompt.
  ctx.on('tools/pre-execute', async (exec, next) => {
    if (exec.name !== 'kernelagent') return next()
    const mode = (exec.arguments as { mode?: unknown } | undefined)?.mode
    if (!requiresPreparedPrompt(mode)) return next()
    const agent = exec.agent
    if (!agent) {
      return { kind: 'deny' as const, reason: 'kernelagent generate/fuse/optimize requires a live agent with a prepared describe prompt' }
    }
    const prepared = peekPreparedPrompt(String(agent.session.id))
    if (!prepared) {
      return {
        kind: 'deny' as const,
        reason: 'Call kernelagent_describe first so Describe1+Describe2 are prepared for this session',
      }
    }
    return next()
  })

  const toolDef = defineTool({
    name: 'kernelagent',
    description: 'Generate, fuse, optimize GPU Triton kernels, or run a predefined example using KernelAgent (from-git). generate/fuse/optimize require a prior kernelagent_describe call.',
    parameters: {
      mode: { type: 'string', required: true, enum: ['generate', 'fuse', 'optimize', 'run_example'], description: 'Mode: generate, fuse, optimize, or run_example.' },
      example_name: { type: 'string', description: 'Example directory name for run_example mode (e.g. optimize_04_musa_sigmoid).' },
      problem_code: { type: 'string', description: 'Ignored for generate/fuse/optimize: the prepared describe prompt is used. Optional for run_example.' },
      initial_kernel: { type: 'string', description: 'Required for optimize mode.' },
      test_code: { type: 'string', description: 'Optional test harness code.' },
      model: { type: 'string', description: 'Optional per-call model override. Omit to use KernelAgent settings.' },
      workers: { type: 'number', description: 'Optional per-call worker override. Omit to use KernelAgent settings.' },
      max_rounds: { type: 'number', description: 'Optional per-call maximum rounds override. Omit to use KernelAgent settings.' },
      platform: { type: 'string', enum: ['cuda', 'musa', 'xpu'], description: 'Optional target platform override. Omit to use KernelAgent settings.' },
      kernel_backend: { type: 'string', enum: ['triton', 'musa'], description: 'Optional code-generation backend override. Omit to use KernelAgent settings.' },
      reasoning_effort: { type: 'string', enum: ['none', 'low', 'medium', 'high', 'xhigh', 'max'], description: 'Optional reasoning effort override. Omit to use KernelAgent settings.' },
      verify: { type: 'boolean', description: 'Optional verification override. Omit to use KernelAgent settings.' },
      enable_experience_memory: { type: 'boolean', description: 'Optional experience-memory override. Omit to use KernelAgent settings.' },
      strategy: { type: 'string', enum: ['beam_search', 'greedy'], description: 'Optional optimization strategy override. Omit to use KernelAgent settings.' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: true,
        properties: {
          success: { type: 'boolean' },
          mode: { type: 'string' },
          report: { type: 'string', description: 'Pre-built markdown report for run_example mode.' },
          perfchart_json: { type: 'string', description: 'JSON string for perfchart Canvas rendering. If present, output a fenced code block with language "perfchart" containing this exact string.' },
          _mergedConfig: { type: 'string', description: 'JSON string of merged global + per-call config for traceability.' },
          kernel_code: {
            oneOf: [
              { type: 'string' },
              { type: 'object', additionalProperties: true, properties: {} },
            ],
          },
          kernel_path: { type: 'string' },
          session_dir: { type: 'string' },
          artifacts_dir: { type: 'string' },
          initial_time_ms: { type: 'number' },
          verification_status: { type: 'string' },
          pytorch_baseline_ms: { type: 'number' },
          best_time_ms: { type: 'number' },
          improvement_pct: { type: 'number' },
          bottleneck: { type: 'string' },
          compute_sol_pct: { type: 'number' },
          memory_sol_pct: { type: 'number' },
          message: { type: 'string' },
        },
      },
      render: (args, value: any) => {
        const lines: string[] = []
        lines.push(`KernelAgent ${value.success ? '✅ succeeded' : '❌ failed'}`)
        if (value.verification_status) lines.push(`Correctness verification: ${value.verification_status}`)
        if (value.optimization_status) lines.push(`Optimization: ${value.optimization_status}`)
        if (value.optimization_error) lines.push(`Optimization error (verified generated kernel retained): ${value.optimization_error}`)
        if (value.total_rounds != null) lines.push(`Optimization rounds: ${value.total_rounds}`)
        if (value.rounds != null) lines.push(`Generation rounds: ${value.rounds}`)
        if (value.kernel_path) lines.push(`Kernel: ${value.kernel_path}`)
        if (value.session_dir) lines.push(`Session: ${value.session_dir}`)
        if (value.artifacts_dir) lines.push(`Artifacts: ${value.artifacts_dir}`)
        if ((args.mode === 'optimize' || value.optimization_status === 'completed') && Number.isFinite(value.initial_time_ms) && value.initial_time_ms > 0
          && Number.isFinite(value.best_time_ms) && value.best_time_ms > 0) {
          lines.push(`Perf: ${value.initial_time_ms.toFixed(3)}ms → ${value.best_time_ms.toFixed(3)}ms`)
          lines.push(`Speedup vs initial kernel: ${(value.initial_time_ms / value.best_time_ms).toFixed(3)}x (below 1 means slower)`)
          if (Number.isFinite(value.pytorch_baseline_ms) && value.pytorch_baseline_ms > 0) {
            lines.push(`PyTorch baseline: ${value.pytorch_baseline_ms}ms; speedup vs PyTorch: ${(value.pytorch_baseline_ms / value.best_time_ms).toFixed(3)}x`)
          }
          if (value.improvement_pct != null) lines.push(`Improvement: ${value.improvement_pct.toFixed(1)}%`)
        }
        if (typeof value.bottleneck === 'string' && value.bottleneck !== '') {
          lines.push(`MCU bottleneck: ${value.bottleneck}`)
        }
        if (Number.isFinite(value.compute_sol_pct) && Number.isFinite(value.memory_sol_pct)) {
          lines.push(`MCU utilization: Compute SOL ${value.compute_sol_pct.toFixed(1)}%, Memory SOL ${value.memory_sol_pct.toFixed(1)}%`)
        }
        if (value.message) lines.push(`Message: ${value.message}`)
        if (value.error && !value.success) lines.push(`Error: ${value.error}`)

        return [{ type: 'text', text: lines.join('\n') }]
      },
      presentationMeta: (args, value: any) => {
        const sourceFiles = value.files ?? value.kernel_code
        const files = typeof sourceFiles === 'object' && sourceFiles !== null
          ? Object.entries(sourceFiles).filter((entry): entry is [string, string] => typeof entry[1] === 'string')
          : typeof sourceFiles === 'string' && sourceFiles !== ''
            ? [[typeof value.kernel_path === 'string' ? value.kernel_path.split(/[\\/]/).pop() || 'kernel.py' : 'kernel.py', sourceFiles]]
            : []
        const timings = [
          { label: 'PyTorch', value: value.pytorch_baseline_ms },
          { label: 'Initial', value: value.initial_time_ms },
          { label: 'Best', value: value.best_time_ms },
        ].filter(item => typeof item.value === 'number' && Number.isFinite(item.value) && item.value > 0)
        const chart = (args.mode === 'optimize' || value.optimization_status === 'completed') && timings.length >= 2
          ? JSON.stringify({ title: 'Kernel Performance', unit: 'ms', data: timings })
          : args.mode === 'run_example' && typeof value.perfchart_json === 'string' ? value.perfchart_json : undefined
        return {
          ...(chart === undefined ? {} : { kernelagentChart: chart }),
          kernelagentFiles: files.map(([fileName, source]) => ({ fileName, source })),
          ...(files.length === 0 && typeof value.report === 'string' ? { kernelagentReport: value.report } : {}),
        }
      },
    },
    async execute(args, exec) {
      const globalConfig = ctx.settings.get(CONFIG_TOOL_SETTINGS_NAMESPACE) as Config | undefined
      if (!globalConfig) throw new Error('KernelAgent settings are not registered')
      const mergedConfig = mergeKernelAgentConfig(args, globalConfig)
      const apiKey = await resolveApiKey(ctx, globalConfig)

      let problemCode = args.problem_code || ''
      if (requiresPreparedPrompt(args.mode)) {
        const agent = exec.agent
        if (!agent) throw new Error('kernelagent generate/fuse/optimize requires a live agent')
        const prepared = takePreparedPrompt(String(agent.session.id))
        if (!prepared) {
          throw new Error('Missing prepared describe prompt; call kernelagent_describe first')
        }
        problemCode = prepared.prompt
        console.log('[kernelagent-tool] ▶ using describe prompt:', {
          describe1Chars: prepared.describe1.length,
          describe2Chars: prepared.describe2.length,
          promptChars: prepared.prompt.length,
        })
      }

      const tmpDir = mkdtempSync(join(tmpdir(), 'ka-'))
      const inputPath = join(tmpDir, 'input.json')
      const outputPath = join(tmpDir, 'output.json')

      // Print merged config for operational traceability / debugging
      console.log('[kernelagent-tool] ▶ merged config:', JSON.stringify(mergedConfig))

      const payload: any = {
        problem_code: problemCode,
        initial_kernel: args.initial_kernel,
        test_code: args.test_code,
        options: {
          mode: args.mode,
          apiKey,
          ...mergedConfig,
        },
      }
      if (args.mode === 'run_example' && args.example_name) {
        payload.example_name = args.example_name
      }
      writeFileSync(inputPath, JSON.stringify(payload), 'utf-8')

      return new Promise((resolve, reject) => {
        const cleanup = () => {
          try { rmSync(tmpDir, { recursive: true, force: true }) } catch (e) { /* ignore */ }
        }

        const proc = spawn(PYTHON, [
          BRIDGE, '--mode', args.mode,
          '--input', inputPath,
          '--output', outputPath,
        ], { cwd: CWD, signal: exec.signal })

        let stderr = ''
        proc.stderr?.on('data', (chunk) => { stderr += chunk })
        proc.on('error', (err) => { cleanup(); reject(err) })
        proc.on('close', (code) => {
          // For run_example, always try to read output.json even if bridge exited non-zero,
          // because bridge may have written fallback results during exception handling.
          if (args.mode === 'run_example' || code === 0) {
            try {
              const output = JSON.parse(readFileSync(outputPath, 'utf-8'))
              // Attach merged config to output so render() can display it in the trajectory
              output._mergedConfig = JSON.stringify(mergedConfig)
              cleanup()
              resolve(output)
              return
            } catch (e: any) {
              cleanup()
              // If reading output.json fails and code is non-zero, report the original error
              if (code !== 0) {
                reject(new Error(`KernelAgent exited ${code} and output.json unreadable: ${e.message}. stderr: ${stderr}`))
                return
              }
              reject(new Error(`Parse error: ${e.message}`))
              return
            }
          }
          cleanup()
          reject(new Error(`KernelAgent exited ${code}. stderr: ${stderr}`))
        })
      })
    },
  })

  ctx.tools.register(toolDef)
  console.log('[kernelagent-tool] registered (from-git)')
}
