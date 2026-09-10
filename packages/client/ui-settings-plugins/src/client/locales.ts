/** Locale bundles for the plugin configuration section and its plugin cards. */

/** Locale keys these surfaces render. */
export type PluginsSettingsLocaleKey =
  | 'nav' | 'title' | 'intro' | 'tabs' | 'configurableTab' | 'empty'
  | 'overridden' | 'reset' | 'readOnly' | 'expand' | 'collapse'
  | 'save' | 'saving' | 'discard' | 'unsaved' | 'saveFailed' | 'invalidNumber' | 'invalidValue'
  | 'bashTitle' | 'bashDescription' | 'bashTimeoutMs' | 'bashTimeoutMsHint'
  | 'bashMaxOutputBytes' | 'bashMaxOutputBytesHint'
  | 'agentLoopTitle' | 'agentLoopDescription' | 'agentLoopMaxParallel' | 'agentLoopMaxParallelHint'
  | 'webSearchTitle' | 'webSearchDescription'
  | 'webSearchApiKey' | 'webSearchApiKeyHint' | 'webSearchApiKeySet' | 'webSearchApiKeyUnset'
  | 'webSearchBaseUrl' | 'webSearchBaseUrlHint' | 'webSearchMaxUses' | 'webSearchMaxUsesHint'
  | 'configToolTitle' | 'configToolDescription'
  | 'configToolApiKey' | 'configToolApiKeyHint'
  | 'configToolModelName' | 'configToolModelNameHint'
  | 'configToolBaseUrl' | 'configToolBaseUrlHint'
  | 'configToolWorkers' | 'configToolWorkersHint'
  | 'configToolMaxRounds' | 'configToolMaxRoundsHint'
  | 'configToolPlatform' | 'configToolPlatformHint'
  | 'configToolKernelBackend' | 'configToolKernelBackendHint'
  | 'configToolStrategy' | 'configToolStrategyHint'
  | 'configToolReasoningEffort' | 'configToolReasoningEffortHint'
  | 'configToolVerify' | 'configToolVerifyHint'
  | 'configToolExperienceMemory' | 'configToolExperienceMemoryHint'

/** English copy. */
export const en: Record<PluginsSettingsLocaleKey, string> = {
  nav: 'Plugins',
  title: 'Plugins',
  intro: 'Configure and inspect the plugins installed in this deployment.',
  tabs: 'Plugin views',
  configurableTab: 'Plugin configuration',
  empty: 'This deployment exposes no plugin settings.',
  overridden: 'Overridden',
  reset: 'Reset to default',
  readOnly: 'This deployment stores settings read-only.',
  expand: 'Show settings',
  collapse: 'Hide settings',
  save: 'Save',
  saving: 'Saving…',
  discard: 'Discard',
  unsaved: 'Unsaved',
  saveFailed: 'The deployment did not accept these values; they were left for you to correct.',
  invalidNumber: 'Enter a number, or leave blank to use the default.',
  invalidValue: 'Choose one of the supported values.',
  bashTitle: 'Shell',
  bashDescription: 'Limits every command the agent runs.',
  bashTimeoutMs: 'Command timeout (ms)',
  bashTimeoutMsHint: 'How long one command may run before it is terminated.',
  bashMaxOutputBytes: 'Output cap per stream (bytes)',
  bashMaxOutputBytesHint: 'Output beyond this spills to a temporary file rather than being lost.',
  agentLoopTitle: 'Agent loop',
  agentLoopDescription: 'How the agent dispatches tool calls.',
  agentLoopMaxParallel: 'Parallel tool calls',
  agentLoopMaxParallelHint: 'Upper bound on parallel-safe calls running at once within one step.',
  webSearchTitle: 'Web search',
  webSearchDescription: 'The DeepSeek search provider.',
  webSearchApiKey: 'API key',
  webSearchApiKeyHint: 'Stored outside the settings file. Leave blank to keep the current key.',
  webSearchApiKeySet: 'A key is configured.',
  webSearchApiKeyUnset: 'No key is configured; search is unavailable until one is.',
  webSearchBaseUrl: 'Endpoint',
  webSearchBaseUrlHint: 'Leave blank to use the provider default.',
  webSearchMaxUses: 'Max searches per request',
  webSearchMaxUsesHint: 'How many times one request may search before it must answer.',
  configToolTitle: 'KernelAgent Tool Config',
  configToolDescription: 'KernelAgent GPU kernel optimizer settings: LLM API key, model, endpoint, and optimization parameters.',
  configToolApiKey: 'API key',
  configToolApiKeyHint: 'Stored in plain text in the settings file for this deployment.',
  configToolModelName: 'Model name',
  configToolModelNameHint: 'Any model id your endpoint accepts, e.g. deepseek-chat, gpt-4o, or grok-4.6.',
  configToolBaseUrl: 'API endpoint',
  configToolBaseUrlHint: 'OpenAI-compatible chat-completions URL.',
  configToolWorkers: 'Workers',
  configToolWorkersHint: 'Parallel optimization threads (1–16).',
  configToolMaxRounds: 'Max rounds',
  configToolMaxRoundsHint: 'Maximum optimization iterations (1–50).',
  configToolPlatform: 'Platform',
  configToolPlatformHint: 'Target GPU platform: cuda / musa / xpu.',
  configToolKernelBackend: 'Kernel backend',
  configToolKernelBackendHint: 'Code generation backend: triton / musa.',
  configToolStrategy: 'Strategy',
  configToolStrategyHint: 'Search strategy: beam_search / greedy.',
  configToolReasoningEffort: 'Reasoning effort',
  configToolReasoningEffortHint: 'OpenAI-compatible effort: none / low / medium / high / xhigh / max.',
  configToolVerify: 'Correctness verification',
  configToolVerifyHint: 'Verify generated output in flows that expose this switch.',
  configToolExperienceMemory: 'Experience memory',
  configToolExperienceMemoryHint: 'Reuse locally stored verified kernel experience.',
}

/** Simplified Chinese copy. */
export const zh: Record<PluginsSettingsLocaleKey, string> = {
  nav: '插件',
  title: '插件',
  intro: '配置和查看本部署已安装的插件。',
  tabs: '插件视图',
  configurableTab: '插件配置',
  empty: '本部署没有开放任何插件设置。',
  overridden: '已覆盖',
  reset: '恢复默认',
  readOnly: '本部署的设置为只读。',
  expand: '展开设置',
  collapse: '收起设置',
  save: '保存',
  saving: '保存中…',
  discard: '放弃修改',
  unsaved: '未保存',
  saveFailed: '本部署没有接受这些值，已保留供你修改。',
  invalidValue: '请选择受支持的选项。',
  invalidNumber: '请填数字；留空表示使用默认值。',
  bashTitle: '终端',
  bashDescription: '限制 agent 运行的每一条命令。',
  bashTimeoutMs: '命令超时（毫秒）',
  bashTimeoutMsHint: '单条命令允许运行多久，超时即终止。',
  bashMaxOutputBytes: '单流输出上限（字节）',
  bashMaxOutputBytesHint: '超出部分会转存到临时文件，而不是被丢弃。',
  agentLoopTitle: 'Agent 循环',
  agentLoopDescription: 'Agent 如何派发工具调用。',
  agentLoopMaxParallel: '并行工具调用数',
  agentLoopMaxParallelHint: '同一步内最多同时运行多少个可并行的调用。',
  webSearchTitle: '网页搜索',
  webSearchDescription: 'DeepSeek 搜索提供方。',
  webSearchApiKey: 'API Key',
  webSearchApiKeyHint: '不写入设置文件。留空表示保持当前密钥。',
  webSearchApiKeySet: '已配置密钥。',
  webSearchApiKeyUnset: '未配置密钥；配置之前搜索不可用。',
  webSearchBaseUrl: '接口地址',
  webSearchBaseUrlHint: '留空则使用提供方默认地址。',
  webSearchMaxUses: '单次请求最多搜索次数',
  webSearchMaxUsesHint: '一次请求在必须作答前最多可以搜索多少次。',
  configToolTitle: 'KernelAgent 工具配置',
  configToolDescription: 'KernelAgent GPU Kernel 优化器配置：LLM API Key、模型、接口地址与优化参数。',
  configToolApiKey: 'API Key',
  configToolApiKeyHint: '以明文保存到本部署的 settings 配置文件。',
  configToolModelName: '模型名称',
  configToolModelNameHint: '接口支持的任意模型 ID，如 deepseek-chat、gpt-4o、grok-4.6。',
  configToolBaseUrl: 'API 地址',
  configToolBaseUrlHint: 'OpenAI 兼容的 chat-completions 接口 URL。',
  configToolWorkers: '工作线程数',
  configToolWorkersHint: '并行优化的工作线程数（1–16）。',
  configToolMaxRounds: '最大轮数',
  configToolMaxRoundsHint: '最多执行多少轮优化迭代（1–50）。',
  configToolPlatform: '目标平台',
  configToolPlatformHint: '目标 GPU 平台：cuda / musa / xpu。',
  configToolKernelBackend: 'Kernel 后端',
  configToolKernelBackendHint: '代码生成后端：triton / musa。',
  configToolStrategy: '搜索策略',
  configToolStrategyHint: '优化搜索策略：beam_search / greedy。',
  configToolReasoningEffort: '推理强度',
  configToolReasoningEffortHint: 'OpenAI 兼容等级：none / low / medium / high / xhigh / max。',
  configToolVerify: '正确性验证',
  configToolVerifyHint: '在支持该开关的流程中验证生成结果。',
  configToolExperienceMemory: '经验记忆',
  configToolExperienceMemoryHint: '复用本地保存且验证通过的 Kernel 经验。',
}
