import type { Context } from '@deepseek-ai/cordis'
import { credentialRef } from '@deepseek-ai/dsh-credentials'
import { installSettingsSection, settingsNamespace } from '@deepseek-ai/dsh-settings'
import { launchEnvironmentOf } from '@deepseek-ai/dsh-launch-environment'
import Schema from '@deepseek-ai/schemastery'
/** Settings namespace for the web configuration card (must match the client card key). */
export const CONFIG_TOOL_SETTINGS_NAMESPACE = settingsNamespace('config-tool')

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
  /** Number of parallel optimization workers (1–16). */
  workers: number
  /** Maximum optimization rounds (1–50). */
  maxRounds: number
  autoOptimize: boolean
  generationMaxRounds: number
  /** Target GPU platform. */
  platform: 'cuda' | 'musa' | 'xpu'
  /** Kernel code-generation backend. */
  kernelBackend: 'triton' | 'musa'
  /** Optimization search strategy. */
  strategy: 'beam_search' | 'greedy'
  /** OpenAI-compatible reasoning effort. */
  reasoningEffort: 'none' | 'low' | 'medium' | 'high' | 'xhigh' | 'max'
  /** Verify outputs in flows that expose a verification switch. */
  verify: boolean
  /** Reuse verified kernels from the local experience store. */
  enableExperienceMemory: boolean
}

/** Schema object exported for Cordis loader validation and UI generation. */
export const ConfigSchema: Schema<Config> = Schema.object({
  apiKey: Schema.string().description('API Key'),
  apiKeyEnv: Schema.string().role('credential-ref').default(DEFAULT_API_KEY_ENV),
  modelName: Schema.string().pattern(/\S/).default('deepseek-chat').description('模型名称'),
  baseURL: Schema.string().pattern(/^https?:\/\//).default('https://api.deepseek.com/v1/chat/completions').description('API 地址'),
  workers: Schema.number().default(4).min(1).max(16).description('并行优化工作线程数'),
  autoOptimize: Schema.boolean().default(false).description('生成后自动优化'),
  generationMaxRounds: Schema.number().default(8).min(1).max(50).description('生成纠错轮数'),
  maxRounds: Schema.number().default(8).min(1).max(50).description('最大优化轮数'),
  platform: Schema.union(['cuda', 'musa', 'xpu']).default('musa').description('目标平台: cuda / musa / xpu'),
  kernelBackend: Schema.union(['triton', 'musa']).default('musa').description('Kernel 后端: triton / musa'),
  strategy: Schema.union(['beam_search', 'greedy']).default('beam_search').description('搜索策略: beam_search / greedy'),
  reasoningEffort: Schema.union(['none', 'low', 'medium', 'high', 'xhigh', 'max']).default('high').description('推理强度'),
  verify: Schema.boolean().default(true).description('启用正确性验证'),
  enableExperienceMemory: Schema.boolean().default(true).description('启用经验记忆'),
})

export const name = 'kernelagent-config-tool'

/**
 * Resolve the API key from the settings section, optional credential ref, or environment.
 * @param ctx - plugin context.
 * @param config - currently authoritative settings section.
 * @returns the key to authorize requests, when one is available.
 */
export async function resolveApiKey(ctx: Context, config: Config): Promise<string | undefined> {
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
  const entry = ConfigSchema(config)
  installSettingsSection(ctx, CONFIG_TOOL_SETTINGS_NAMESPACE, ConfigSchema, entry, {
    setSource: () => {},
    onChange: () => {},
  })
}
