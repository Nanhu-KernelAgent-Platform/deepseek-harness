import { defineTool } from '@deepseek-ai/dsh-tools'
import type { Context } from '@deepseek-ai/cordis'
import { credentialRef } from '@deepseek-ai/dsh-credentials'
import { installSettingsSection, settingsNamespace } from '@deepseek-ai/dsh-settings'
import { launchEnvironmentOf } from '@deepseek-ai/dsh-launch-environment'
import Schema from '@deepseek-ai/schemastery'
import { writeFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

// ESM-compatible __dirname polyfill (DSH loader runs files in ESM scope)
const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

/** Settings namespace for the web configuration card (must match the client card key). */
export const CONFIG_TOOL_SETTINGS_NAMESPACE = settingsNamespace('config-tool')

/** Shared file used as cross-plugin communication channel for global config. */
export const KERNELAGENT_CONFIG_FILE = resolve(__dirname, 'config.json')

const DEFAULT_API_KEY_ENV = 'OPENAI_API_KEY'

/** Plugin config editable from cordis.yml and the web settings card. */
export interface Config {
  /** LLM API key; stored in the settings file. */
  apiKey?: string
  /** Credential reference resolved for each call. */
  apiKeyEnv?: string
  /** LLM model id (DeepSeek, OpenAI, or any compatible provider). */
  modelName: string
  /** Chat-completions endpoint (OpenAI-compatible). */
  baseURL: string
  /** Sequential API calls per tool invocation (1–10). */
  iterations: number
  /** Number of parallel optimization workers (1–16). */
  workers: number
  /** Maximum optimization rounds (1–50). */
  maxRounds: number
  /** Target GPU platform: cuda / musa / rocm. */
  platform: string
  /** Kernel backend: triton / cuda. */
  kernelBackend: string
  /** Optimization strategy: beam_search / random_walk. */
  strategy: string
}

/** Schema object exported for Cordis loader validation and UI generation. */
export const ConfigSchema: Schema<Config> = Schema.object({
  apiKey: Schema.string().description('API Key'),
  apiKeyEnv: Schema.string().role('credential-ref').default(DEFAULT_API_KEY_ENV),
  modelName: Schema.string().default('deepseek-chat').description('模型名称'),
  baseURL: Schema.string().default('https://api.deepseek.com/v1/chat/completions').description('API 地址'),
  iterations: Schema.number().default(1).min(1).max(10).description('迭代次数（1~10）'),
  workers: Schema.number().default(4).min(1).max(16).description('并行优化工作线程数'),
  maxRounds: Schema.number().default(8).min(1).max(50).description('最大优化轮数'),
  platform: Schema.string().default('musa').description('目标平台: cuda / musa / rocm'),
  kernelBackend: Schema.string().default('triton').description('kernel 后端: triton / cuda'),
  strategy: Schema.string().default('beam_search').description('搜索策略: beam_search / random_walk'),
})

export const name = 'kernelagent-config-tool'

/**
 * Resolve the API key from the settings section, optional credential ref, or environment.
 * @param ctx - plugin context.
 * @param config - currently authoritative settings section.
 * @returns the key to authorize requests, when one is available.
 */
async function resolveApiKey(ctx: Context, config: Config): Promise<string | undefined> {
  if (config.apiKey !== undefined && config.apiKey.length > 0) return config.apiKey
  const ref = credentialRef(config.apiKeyEnv ?? DEFAULT_API_KEY_ENV)
  const credentials = ctx.get('credentials')
  if (credentials !== undefined) {
    const resolved = await credentials.resolve(ref)
    if (resolved?.value !== undefined && resolved.value.length > 0) return resolved.value
  }
  const ambient = launchEnvironmentOf(ctx).get(ref)
  return ambient !== undefined && ambient.value.length > 0 ? ambient.value : undefined
}

export function apply(ctx: Context, config?: Config) {
  const defaults: Config = {
    apiKey: '',
    apiKeyEnv: DEFAULT_API_KEY_ENV,
    modelName: 'deepseek-chat',
    baseURL: 'https://api.deepseek.com/v1/chat/completions',
    iterations: 1,
    workers: 4,
    maxRounds: 8,
    platform: 'musa',
    kernelBackend: 'triton',
    strategy: 'beam_search',
  }
  let current: () => Config = () => config ?? defaults

  const persist = () => {
    try {
      const cfg = current?.()
      if (!cfg) return
      // SECURITY: exclude apiKey from persistent config — it is resolved at runtime
      // via DSH credentials (credentialRef + ctx.credentials.resolve()) or environment.
      const { apiKey, ...safeConfig } = cfg
      writeFileSync(KERNELAGENT_CONFIG_FILE, JSON.stringify(safeConfig, null, 2), 'utf-8')
    } catch (e) {
      console.warn('[config-tool] failed to persist config file:', e)
    }
  }

  installSettingsSection(ctx, CONFIG_TOOL_SETTINGS_NAMESPACE, ConfigSchema, current(), {
    setSource: (source) => { current = source },
    onChange: () => {
      console.log('[config-tool] settings changed:', current())
      persist()
    },
  })

  // Persist initial config immediately so that kernelagent-tool can read it
  // even before any user edit through the web card.
  persist()

  console.log('✅ Config-tool 加载成功，模型：', current().modelName, '迭代次数：', current().iterations)
}
