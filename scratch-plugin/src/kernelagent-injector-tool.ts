import { defineTool } from '@deepseek-ai/dsh-tools'
import type { Context } from '@deepseek-ai/cordis'
import { spawn } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { CONFIG_TOOL_SETTINGS_NAMESPACE, type Config } from './kernelagent-config-tool.ts'

// ========== Runtime configuration via environment variables ==========
// These are injected by start_dsh.sh or the host environment.
const PYTHON = process.env.KERNELAGENT_PYTHON || 'python3'

function requiredEnvironment(name: string): string {
  const value = process.env[name]
  if (!value) {
    throw new Error(
      `[kernelagent-injector-tool] Missing required environment variable: ${name}. ` +
        `Set it in start_dsh.sh (see KERNELAGENT_INJECTOR_BRIDGE / KERNELAGENT_WORKING_DIR).`,
    )
  }
  return value
}

const INJECTOR_BRIDGE = requiredEnvironment('KERNELAGENT_INJECTOR_BRIDGE')
const CWD = requiredEnvironment('KERNELAGENT_WORKING_DIR')

export const name = 'kernelagent-injector-tool'
export const inject = ['tools', 'systemPrompt', 'settings']

export function apply(ctx: Context) {
  ctx.systemPrompt.section({
    name: 'tool:kernelagent-injector',
    order: 215,
      text: `Use the kernelagent_inject tool (batch-2 dynamic flow) to deploy a KernelAgent-generated kernel bundle (best_bundle) into the managed runtime store, so training code can call it via torch.ops.kernelagent::<op>(x).
Input "source" accepts: an example name (e.g. optimize_06_musa_relu, the latest run is picked automatically), a run directory, a best_bundle absolute path, a single-file kernel (triton) or a serialized payload file ("FILE: ..." blocks).
R1 real-generation priority: resolver prefers results/run_* (real optimized kernel) over baseline_bundle/ (fallback). If the example has no run_* results but has a baseline_bundle/, the baseline is used as fallback (manifest marks source.kind=baseline). Use allow_baseline=false to enforce real optimized kernel and prevent deploying unoptimized baseline (the guard applies to every source form, including a directory or best_bundle path).
R2 source.kind is provenance-based, never guessed from a directory name: "run" = a results/run_* bundle, "baseline" = an examples/<example>/baseline_bundle or a directory literally named baseline_bundle, "local" = a caller-supplied bundle that is neither (it is NOT an unoptimized baseline, so allow_baseline does not apply to it). status for a "local" bundle is read from its own provenance metadata (demo_summary.json / result.json / _meta.json / _provenance.json, following a result_json pointer to the real generation result); "UNKNOWN" means no provenance file was found. Report source.kind and status as returned — do not restate a "local" bundle as a baseline.
R3 deploy_dir: defaults to <active Workspace>/.kernelagent/runtime. A saved or per-call custom directory overrides that default. The directory is bootstrapped as a self-contained store (kernelagent_runtime package auto-copied if missing). Activation hint points to deploy_dir.
When the user names the operator to replace, pass op_name aligned with the target training op (e.g. replacing F.relu -> op_name=relu).
By default (verify=true, the REQUIRED flow) the tool also runs the real training task to confirm the operator is actually integrated. R5 dynamic verify step generation uses four priorities: (P1) verify_cmd user command, (P2) train_script-derived steps from R6-generated injected script, (P3) <train_dir>/verify_tasks.json convention file, (P4) built-in defaults for known ops. The report contains an "integration verify" section with PASS/FAIL per step, the kernel call count (proof the injected path was taken), and an overall verdict. Only report "integrated/verified" when the verify section is PASS; deploy alone is NOT sufficient.
R6 train_script: when provided, the bridge auto-generates an injected training script (<stem>_ka_injected.py) by AST-analyzing the original and inserting a kernelagent-runtime scaffold that patches torch.nn.functional and torch module attributes (alias-proof). The injected script is then used as the P2 verify train step.
After a successful deploy+verify, also tell the user how to re-run it manually (the activation_hint field), e.g.:
  export PYTHONPATH=<runtime_store>:$PYTHONPATH
  python -c "import kernelagent_runtime, torch; print(torch.ops.kernelagent.<op>)"
Output the report field verbatim, including any code fences.`,
  })

  const toolDef = defineTool({
    name: 'kernelagent_inject',
    description:
      'Deploy a KernelAgent best_bundle into the runtime store and register it as ' +
      'torch.ops.kernelagent::<op> (batch-2 dynamic flow: deploy_dir, three-layer manifest, real-generation priority).',
    parameters: {
      source: {
        type: 'string',
        required: true,
        description:
          'example name (e.g. optimize_06_musa_relu) | run dir | best_bundle dir | single-file kernel | serialized payload file',
      },
      op_name: {
        type: 'string',
        description:
          'Registered op name, aligned with the target training op it replaces (e.g. relu). ' +
          'Defaults to deriving from problem.py reference implementation when available.',
      },
      build: {
        type: 'boolean',
        description: 'Whether to run "setup.py build_ext --inplace" for musa bundles (default true).',
      },
      verify: {
        type: 'boolean',
        description:
          'Whether to run the real training task from KERNELAGENT_TRAIN_DIR to verify integration (default true, required flow). ' +
          'Set false only to skip the integration check.',
      },
      allow_baseline: {
        type: 'boolean',
        description:
          'Allow fallback to baseline_bundle when no run_* results exist (default true). ' +
          'Set false to enforce real optimized kernel generation and prevent deploying unoptimized baseline.',
      },
      deploy_dir: {
        type: 'string',
        description:
          'User-specified deployment directory (default = <active Workspace>/.kernelagent/runtime). ' +
          'The directory is bootstrapped as a self-contained store (kernelagent_runtime package copied if missing). ' +
          'Activation hint will point to this directory.',
      },
      trace: {
        type: 'boolean',
        description:
          'Whether to emit a detailed trace of the resolution and deploy steps in the report (default true). ' +
          'Trace is also persisted in the L3 history JSON.',
      },
      verify_cmd: {
        type: 'string',
        description:
          'R5 P1: User-provided verify command(s). Highest priority. ' +
          'A single command string or JSON array of commands. ' +
          'If given, the bridge runs these commands directly (cwd = script dir or KERNELAGENT_TRAIN_DIR) ' +
          'and checks exit code + kernel_calls counter + optional verify_expect.',
      },
      verify_expect: {
        type: 'string',
        description:
          'R5 P1: Expected output marker for user-provided verify_cmd (e.g. "acc=9"). ' +
          'Checked in addition to exit code and kernel_calls counter.',
      },
      train_script: {
        type: 'string',
        description:
          'R5 P2 / R6: Path to the user training script (e.g. /path/train_mnist_musa.py). ' +
          'When provided, the bridge generates an injected version (<stem>_ka_injected.py) and ' +
          'derives verify steps from it (numeric_check + train).',
      },
      train_args: {
        type: 'string',
        description:
          'Optional extra arguments passed to the injected training script ' +
          '(e.g. "--epochs 1 --batch_size 64"). Only used when train_script is provided.',
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: true,
        properties: {
          success: { type: 'boolean', description: 'Whether the deploy (build) succeeded.' },
          verified: { type: 'boolean', description: 'Whether the integration verify (real training task) passed.' },
          report: { type: 'string', description: 'Pre-built markdown report for the user.' },
          activation_hint: { type: 'string', description: 'How to activate in the training environment.' },
          message: { type: 'string' },
        },
      },
      render: (_args: any, value: any) => {
        const blocks: { type: 'text'; text: string }[] = []
        if (typeof value.report === 'string' && value.report.length > 0) {
          blocks.push({ type: 'text', text: value.report })
        } else {
          blocks.push({ type: 'text', text: value.message ?? JSON.stringify(value, null, 2) })
        }
        return blocks
      },
    },
    async execute(args: any, exec: any) {
      const globalConfig = ctx.settings.get(CONFIG_TOOL_SETTINGS_NAMESPACE) as Config | undefined
      const workspaceDir = exec?.agent?.session.header.cwd
      const tmpDir = mkdtempSync(join(tmpdir(), 'kainj-'))
      const inputPath = join(tmpDir, 'input.json')
      const outputPath = join(tmpDir, 'output.json')

      const payload: Record<string, unknown> = {
        source: args.source,
        op_name: args.op_name || globalConfig?.injectOpName || undefined,
        build: args.build,
        verify: args.verify ?? globalConfig?.injectVerify,
        allow_baseline: args.allow_baseline,
        workspace_dir: workspaceDir,
        deploy_dir: args.deploy_dir || globalConfig?.injectDeployDir || undefined,
        trace: args.trace,
        verify_cmd: args.verify_cmd,
        verify_expect: args.verify_expect,
        train_script: args.train_script || globalConfig?.injectTrainScript || undefined,
        train_args: args.train_args || globalConfig?.injectTrainArgs || undefined,
      }
      writeFileSync(inputPath, JSON.stringify(payload), 'utf-8')

      return new Promise((resolve, reject) => {
        const cleanup = () => {
          try {
            rmSync(tmpDir, { recursive: true, force: true })
          } catch {
            /* ignore cleanup errors */
          }
        }

        const proc = spawn(
          PYTHON,
          [INJECTOR_BRIDGE, '--mode', 'deploy', '--input', inputPath, '--output', outputPath],
          { cwd: CWD, signal: exec?.signal },
        )

        let stderr = ''
        proc.stderr?.on('data', (chunk) => {
          stderr += chunk
        })
        proc.on('error', (err) => {
          cleanup()
          reject(err)
        })
        proc.on('close', (code) => {
          if (code === 0) {
            try {
              const output = JSON.parse(readFileSync(outputPath, 'utf-8'))
              cleanup()
              resolve(output)
              return
            } catch (e: any) {
              cleanup()
              reject(new Error(`injector bridge output unreadable: ${e.message}`))
              return
            }
          }
          cleanup()
          reject(new Error(`injector bridge exited ${code}. stderr: ${stderr}`))
        })
      })
    },
  })

  ctx.tools.register(toolDef)
  console.log('[kernelagent-injector-tool] registered (batch-1 MVP)')
}
