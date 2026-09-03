import { defineTool } from '@deepseek-ai/dsh-tools'
import type { Context } from '@deepseek-ai/cordis'
import { spawn } from 'node:child_process'
import { writeFileSync, readFileSync, mkdtempSync, existsSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

// ESM-compatible __dirname polyfill (DSH loader runs files in ESM scope)
const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// ========== Runtime configuration via environment variables ==========
// These are injected by start_dsh.sh or the host environment.
const PYTHON = process.env.KERNELAGENT_PYTHON || 'python3'
const BRIDGE = process.env.KERNELAGENT_BRIDGE
const CWD    = process.env.KERNELAGENT_WORKING_DIR

if (!BRIDGE) {
  throw new Error(
    '[kernelagent-tool] Missing required environment variable: KERNELAGENT_BRIDGE.\n' +
    'Please set it to the absolute path of kernelagent_bridge.py (e.g. /path/to/KernelAgent-from-git/kernelagent_bridge.py).'
  )
}
if (!CWD) {
  throw new Error(
    '[kernelagent-tool] Missing required environment variable: KERNELAGENT_WORKING_DIR.\n' +
    'Please set it to the absolute path of the KernelAgent project root.'
  )
}

/** Shared configuration file written by kernelagent-config-tool (non-sensitive fields only). */
const KERNELAGENT_CONFIG_FILE = resolve(__dirname, 'config.json')

/** Resolve API key from environment variables (secure, no file-based key storage). */
function resolveApiKey(): string {
  return process.env.KERNELAGENT_API_KEY
    || process.env.OPENAI_API_KEY
    || process.env.DEEPSEEK_API_KEY
    || ''
}

export const name = 'kernelagent-tool'
export const inject = ['tools', 'systemPrompt']

export function apply(ctx: Context) {
  ctx.systemPrompt.section({
    name: 'tool:kernelagent',
    order: 210,
    text: `Use the kernelagent tool to generate, fuse, optimize GPU Triton kernels, or run a predefined example.
Modes: generate | fuse | optimize | run_example
- run_example: runs a predefined example (e.g. optimize_04_musa_sigmoid) and returns its artifacts.

For run_example, your reply MUST contain BOTH of the following, in this order:
1. A SHORT SUMMARY paragraph (in the user's language): run status (success/fallback), performance metrics (initial → best time, improvement), and 2-3 key observations about the generated kernel (e.g. vectorization strategy, memory access pattern).
2. The COMPLETE code: output the entire contents of the "report" field VERBATIM, including all code blocks delimited by triple backticks (kernel.mu, kernel.py, binding.cpp, setup.py, optimized_kernel_musa.py). Do NOT summarize, paraphrase, truncate, or skip any code. Do not write "I will show the code" — just paste the report content after the summary so the user sees both the analysis and the actual generated kernel code in the chat.
3. Performance chart: if the result contains a "perfchart_json" field, output a fenced code block with language "perfchart" and put the raw JSON string from "perfchart_json" inside it, like this:
\`\`\`perfchart
<the exact value of result.perfchart_json>
\`\`\`
The web UI will automatically render this as an interactive bar chart comparing PyTorch / Initial / Best kernel latency. Do NOT modify the JSON content — paste it exactly as provided.`,
  })

  const toolDef = defineTool({
    name: 'kernelagent',
    description: 'Generate, fuse, optimize GPU Triton kernels, or run a predefined example using KernelAgent (from-git).',
    parameters: {
      mode: { type: 'string', required: true, description: 'Mode: generate, fuse, optimize, or run_example.' },
      example_name: { type: 'string', description: 'Example directory name for run_example mode (e.g. optimize_04_musa_sigmoid).' },
      problem_code: { type: 'string', description: 'PyTorch code or description. Required for generate/fuse/optimize.' },
      initial_kernel: { type: 'string', description: 'Required for optimize mode.' },
      test_code: { type: 'string', description: 'Optional test harness code.' },
      model: { type: 'string', default: 'gpt-5' },
      workers: { type: 'number', default: 4 },
      max_rounds: { type: 'number', default: 8 },
      platform: { type: 'string', default: 'cuda' },
      kernel_backend: { type: 'string', default: 'triton' },
      verify: { type: 'boolean', default: true },
      enable_experience_memory: { type: 'boolean', default: true },
      strategy: { type: 'string', default: 'beam_search' },
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
          kernel_code: { type: 'string' },
          kernel_path: { type: 'string' },
          session_dir: { type: 'string' },
          artifacts_dir: { type: 'string' },
          initial_time_ms: { type: 'number' },
          best_time_ms: { type: 'number' },
          improvement_pct: { type: 'number' },
          message: { type: 'string' },
        },
      },
      render: (_args, value: any) => {
        // Format merged config for display at the very top of the tool output
        let mergedConfigDisplay = ''
        if (value._mergedConfig) {
          let cfgObj: any = value._mergedConfig
          if (typeof cfgObj === 'string') cfgObj = JSON.parse(cfgObj)
          const kv = Object.entries(cfgObj as Record<string, any>)
            .map(([k, v]) => `${k}: ${v}`)
            .join(' | ')
          mergedConfigDisplay = `━━━ KernelAgent Execution Config ━━━\n${kv}\n\n`
        }

        if (value.mode === 'run_example' && typeof value.report === 'string') {
          const blocks: any[] = [
            { type: 'text', text: mergedConfigDisplay + value.report },
          ]
          // ALWAYS append perfchart block for run_example mode (demo data for now)
          const chartJson = value.perfchart_json ?? JSON.stringify({
            title: 'Kernel Performance',
            unit: 'ms',
            data: [
              { label: 'PyTorch', value: 0.00822, color: '#5470c6' },
              { label: 'Initial', value: 0.00679, color: '#91cc75' },
              { label: 'Best', value: 0.00543, color: '#fac858' },
            ],
            improvement_pct: 20.0,
            is_mock: true,
            note: 'Demo data.',
          }, null, 2)
          blocks.push({ type: 'text', text: '\n\n━━━ Performance Chart (Demo Data) ━━━\n' })
          blocks.push({ type: 'text', text: '```perfchart\n' + chartJson + '\n```' })

          // Append merged config JSON for traceability (visible in Trajectory Output tab)
          if (value._mergedConfig) {
            let cfgObj: any = value._mergedConfig
            if (typeof cfgObj === 'string') cfgObj = JSON.parse(cfgObj)
            const cfgText = JSON.stringify(cfgObj, null, 2)
            blocks.push({ type: 'text', text: '\n\n<!-- tool-config -->\n```json\n"merged_config": ' + cfgText + '\n```\n' })
          }

          return blocks
        }
        const lines: string[] = []
        lines.push(mergedConfigDisplay)
        lines.push(`KernelAgent ${value.success ? '✅ succeeded' : '❌ failed'}`)
        if (value.kernel_path) lines.push(`Kernel: ${value.kernel_path}`)
        if (value.session_dir) lines.push(`Session: ${value.session_dir}`)
        if (value.artifacts_dir) lines.push(`Artifacts: ${value.artifacts_dir}`)
        if (value.initial_time_ms != null && value.best_time_ms != null) {
          lines.push(`Perf: ${value.initial_time_ms.toFixed(3)}ms → ${value.best_time_ms.toFixed(3)}ms`)
          if (value.improvement_pct != null) lines.push(`Improvement: ${value.improvement_pct.toFixed(1)}%`)
        }
        if (value.message) lines.push(`Message: ${value.message}`)
        if (value.error && !value.success) lines.push(`Error: ${value.error}`)

        return [{ type: 'text', text: lines.join('\n') }]
      },
    },
    async execute(args, exec) {
      const tmpDir = mkdtempSync(join(tmpdir(), 'ka-'))
      const inputPath = join(tmpDir, 'input.json')
      const outputPath = join(tmpDir, 'output.json')

      // 读取 Web 配置卡片中的全局配置（由 kernelagent-config-tool 写入）
      let globalConfig: any = {}
      try {
        if (existsSync(KERNELAGENT_CONFIG_FILE)) {
          globalConfig = JSON.parse(readFileSync(KERNELAGENT_CONFIG_FILE, 'utf-8'))
          console.log('[kernelagent-tool] loaded global config from', KERNELAGENT_CONFIG_FILE)
        }
      } catch (e: any) {
        console.warn('[kernelagent-tool] failed to read global config:', e.message)
      }

      // Build merged config with priority: args > globalConfig > defaults
      // SECURITY: apiKey is resolved from environment variables, never from disk.
      const mergedConfig = {
        model: args.model ?? globalConfig.modelName ?? 'deepseek-chat',
        workers: args.workers ?? globalConfig.workers ?? 4,
        max_rounds: args.max_rounds ?? globalConfig.maxRounds ?? 8,
        verify: args.verify ?? globalConfig.verify ?? true,
        platform: args.platform ?? globalConfig.platform ?? 'musa',
        kernel_backend: args.kernel_backend ?? globalConfig.kernelBackend ?? 'triton',
        enable_experience_memory: args.enable_experience_memory ?? globalConfig.enable_experience_memory ?? true,
        strategy: args.strategy ?? globalConfig.strategy ?? 'beam_search',
        apiKey: resolveApiKey(),
        baseURL: globalConfig.baseURL ?? 'https://api.deepseek.com/v1/chat/completions',
        iterations: globalConfig.iterations ?? 1,
      }

      // Print merged config for operational traceability / debugging
      console.log('[kernelagent-tool] ▶ merged config:', JSON.stringify(mergedConfig))

      const payload: any = {
        problem_code: args.problem_code || '',
        initial_kernel: args.initial_kernel,
        test_code: args.test_code,
        options: {
          mode: args.mode,
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
