/** The KernelAgent config card: API key, model name, API endpoint, and optimization params. */

import type { InjectFace, PropsLocale, PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import { BooleanField, SelectField, ValueField } from './fields.tsx'
import { PluginCard } from './PluginCard.tsx'
import type { ConfigToolCardFace } from './kernelagent-config-card-controller.ts'

export type ConfigToolCardProps =
  PropsRuntime<'settings.plugin.item'>
  & PropsLocale<'settings.plugins'>
  & InjectFace<ConfigToolCardFace>

/** Render the KernelAgent config card. */
export function KernelAgentConfigCard(props: ConfigToolCardProps) {
  const { t } = props
  const state = props.useConfigToolCard(snapshot => snapshot)
  const disabled = !state.writable
  return (
    <PluginCard
      t={t}
      titleKey="configToolTitle"
      descriptionKey="configToolDescription"
      state={state}
      onSave={props.save}
      onDiscard={props.discard}
    >
      <ValueField
        id="plugin-config-config-tool-api-key"
        label={t('configToolApiKey')}
        hint={t('configToolApiKeyHint')}
        overriddenLabel={t('overridden')}
        resetLabel={t('reset')}
        invalidLabel={t('invalidNumber')}
        disabled={disabled}
        {...state.apiKey}
        onEdit={(text) => { props.edit('apiKey', text) }}
        onReset={() => { props.resetField('apiKey') }}
      />
      <ValueField
        id="plugin-config-config-tool-model"
        label={t('configToolModelName')}
        hint={t('configToolModelNameHint')}
        overriddenLabel={t('overridden')}
        resetLabel={t('reset')}
        invalidLabel={t('invalidValue')}
        disabled={disabled}
        placeholder="deepseek-chat"
        {...state.modelName}
        onEdit={(text) => { props.edit('modelName', text) }}
        onReset={() => { props.resetField('modelName') }}
      />
      <ValueField
        id="plugin-config-config-tool-base-url"
        label={t('configToolBaseUrl')}
        hint={t('configToolBaseUrlHint')}
        overriddenLabel={t('overridden')}
        resetLabel={t('reset')}
        invalidLabel={t('invalidValue')}
        disabled={disabled}
        placeholder="https://api.deepseek.com/v1/chat/completions"
        {...state.baseURL}
        onEdit={(text) => { props.edit('baseURL', text) }}
        onReset={() => { props.resetField('baseURL') }}
      />
      <ValueField
        id="plugin-config-config-tool-workers"
        label={t('configToolWorkers')}
        hint={t('configToolWorkersHint')}
        overriddenLabel={t('overridden')}
        resetLabel={t('reset')}
        invalidLabel={t('invalidNumber')}
        numeric
        disabled={disabled}
        {...state.workers}
        onEdit={(text) => { props.edit('workers', text) }}
        onReset={() => { props.resetField('workers') }}
      />
      <ValueField
        id="plugin-config-config-tool-max-rounds"
        label={t('configToolMaxRounds')}
        hint={t('configToolMaxRoundsHint')}
        overriddenLabel={t('overridden')}
        resetLabel={t('reset')}
        invalidLabel={t('invalidNumber')}
        numeric
        disabled={disabled}
        {...state.maxRounds}
        onEdit={(text) => { props.edit('maxRounds', text) }}
        onReset={() => { props.resetField('maxRounds') }}
      />
      <SelectField
        id="plugin-config-config-tool-platform"
        label={t('configToolPlatform')}
        hint={t('configToolPlatformHint')}
        overriddenLabel={t('overridden')}
        resetLabel={t('reset')}
        invalidLabel={t('invalidValue')}
        disabled={disabled}
        options={['cuda', 'musa', 'xpu']}
        {...state.platform}
        onEdit={(text) => { props.edit('platform', text) }}
        onReset={() => { props.resetField('platform') }}
      />
      <SelectField
        id="plugin-config-config-tool-kernel-backend"
        label={t('configToolKernelBackend')}
        hint={t('configToolKernelBackendHint')}
        overriddenLabel={t('overridden')}
        resetLabel={t('reset')}
        invalidLabel={t('invalidValue')}
        disabled={disabled}
        options={['triton', 'musa']}
        {...state.kernelBackend}
        onEdit={(text) => { props.edit('kernelBackend', text) }}
        onReset={() => { props.resetField('kernelBackend') }}
      />
      <SelectField
        id="plugin-config-config-tool-strategy"
        label={t('configToolStrategy')}
        hint={t('configToolStrategyHint')}
        overriddenLabel={t('overridden')}
        resetLabel={t('reset')}
        invalidLabel={t('invalidValue')}
        disabled={disabled}
        options={['beam_search', 'greedy']}
        {...state.strategy}
        onEdit={(text) => { props.edit('strategy', text) }}
        onReset={() => { props.resetField('strategy') }}
      />
      <SelectField
        id="plugin-config-config-tool-reasoning-effort"
        label={t('configToolReasoningEffort')}
        hint={t('configToolReasoningEffortHint')}
        overriddenLabel={t('overridden')}
        resetLabel={t('reset')}
        invalidLabel={t('invalidValue')}
        disabled={disabled}
        options={['none', 'low', 'medium', 'high', 'xhigh', 'max']}
        {...state.reasoningEffort}
        onEdit={(text) => { props.edit('reasoningEffort', text) }}
        onReset={() => { props.resetField('reasoningEffort') }}
      />
      <BooleanField
        id="plugin-config-config-tool-verify"
        label={t('configToolVerify')}
        hint={t('configToolVerifyHint')}
        overriddenLabel={t('overridden')}
        resetLabel={t('reset')}
        disabled={disabled}
        {...state.verify}
        onEdit={(text) => { props.edit('verify', text) }}
        onReset={() => { props.resetField('verify') }}
      />
      <BooleanField
        id="plugin-config-config-tool-experience-memory"
        label={t('configToolExperienceMemory')}
        hint={t('configToolExperienceMemoryHint')}
        overriddenLabel={t('overridden')}
        resetLabel={t('reset')}
        disabled={disabled}
        {...state.enableExperienceMemory}
        onEdit={(text) => { props.edit('enableExperienceMemory', text) }}
        onReset={() => { props.resetField('enableExperienceMemory') }}
      />
    </PluginCard>
  )
}
