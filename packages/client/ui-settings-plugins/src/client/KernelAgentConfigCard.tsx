/** The KernelAgent config card: API key, model name, API endpoint, and optimization params. */

import type { InjectFace, PropsLocale, PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import { ValueField } from './fields.tsx'
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
        invalidLabel={t('invalidNumber')}
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
        invalidLabel={t('invalidNumber')}
        disabled={disabled}
        placeholder="https://api.deepseek.com/v1/chat/completions"
        {...state.baseURL}
        onEdit={(text) => { props.edit('baseURL', text) }}
        onReset={() => { props.resetField('baseURL') }}
      />
      <ValueField
        id="plugin-config-config-tool-iterations"
        label={t('configToolIterations')}
        hint={t('configToolIterationsHint')}
        overriddenLabel={t('overridden')}
        resetLabel={t('reset')}
        invalidLabel={t('invalidNumber')}
        numeric
        disabled={disabled}
        {...state.iterations}
        onEdit={(text) => { props.edit('iterations', text) }}
        onReset={() => { props.resetField('iterations') }}
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
      <ValueField
        id="plugin-config-config-tool-platform"
        label={t('configToolPlatform')}
        hint={t('configToolPlatformHint')}
        overriddenLabel={t('overridden')}
        resetLabel={t('reset')}
        invalidLabel={t('invalidNumber')}
        disabled={disabled}
        placeholder="musa"
        {...state.platform}
        onEdit={(text) => { props.edit('platform', text) }}
        onReset={() => { props.resetField('platform') }}
      />
      <ValueField
        id="plugin-config-config-tool-kernel-backend"
        label={t('configToolKernelBackend')}
        hint={t('configToolKernelBackendHint')}
        overriddenLabel={t('overridden')}
        resetLabel={t('reset')}
        invalidLabel={t('invalidNumber')}
        disabled={disabled}
        placeholder="triton"
        {...state.kernelBackend}
        onEdit={(text) => { props.edit('kernelBackend', text) }}
        onReset={() => { props.resetField('kernelBackend') }}
      />
      <ValueField
        id="plugin-config-config-tool-strategy"
        label={t('configToolStrategy')}
        hint={t('configToolStrategyHint')}
        overriddenLabel={t('overridden')}
        resetLabel={t('reset')}
        invalidLabel={t('invalidNumber')}
        disabled={disabled}
        placeholder="beam_search"
        {...state.strategy}
        onEdit={(text) => { props.edit('strategy', text) }}
        onReset={() => { props.resetField('strategy') }}
      />
    </PluginCard>
  )
}
