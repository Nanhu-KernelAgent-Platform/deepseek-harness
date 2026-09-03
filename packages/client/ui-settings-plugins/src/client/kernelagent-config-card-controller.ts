/** The config-tool card's staged form over the `config-tool` settings namespace. */

import type { SettingsScope, SnapshotStore } from '@deepseek-ai/dsh-client-runtime/client'
import {
  CardForm, numberField, textField,
  type CardActions, type CardFieldState, type CardShell,
} from './card-form.ts'

/** Namespace of the config-tool plugin. */
export const CONFIG_TOOL_NS = 'config-tool'

/** The config-tool fields this card edits (persisted in the settings document). */
export interface ConfigToolSettings {
  /** LLM API key; stored in the settings file under this namespace. */
  apiKey?: string
  modelName?: string
  baseURL?: string
  iterations?: number
  workers?: number
  maxRounds?: number
  platform?: string
  kernelBackend?: string
  strategy?: string
}

/** What the config-tool card renders. */
export interface ConfigToolCardState extends CardShell {
  apiKey: CardFieldState
  modelName: CardFieldState
  baseURL: CardFieldState
  iterations: CardFieldState
  workers: CardFieldState
  maxRounds: CardFieldState
  platform: CardFieldState
  kernelBackend: CardFieldState
  strategy: CardFieldState
}

/** The registration-side face the config-tool card's slot entry injects. */
export interface ConfigToolCardFace extends CardActions {
  hooks: {
    configToolCard: SnapshotStore<ConfigToolCardState>
  }
}

/** Bridges the `config-tool` scope onto the card's staged form. */
export class ConfigToolCardController {
  private readonly form: CardForm<ConfigToolSettings>
  private readonly store: SnapshotStore<ConfigToolCardState>

  /** @param scope - the bound settings scope for the `config-tool` namespace. */
  constructor(scope: SettingsScope<ConfigToolSettings>) {
    this.form = new CardForm(scope, [
      textField('apiKey'),
      textField('modelName'),
      textField('baseURL'),
      numberField('iterations'),
      numberField('workers'),
      numberField('maxRounds'),
      textField('platform'),
      textField('kernelBackend'),
      textField('strategy'),
    ])
    this.store = this.form.bind(() => this.projection())
  }

  private projection(): ConfigToolCardState {
    return {
      ...this.form.shell(),
      apiKey: this.form.field('apiKey'),
      modelName: this.form.field('modelName'),
      baseURL: this.form.field('baseURL'),
      iterations: this.form.field('iterations'),
      workers: this.form.field('workers'),
      maxRounds: this.form.field('maxRounds'),
      platform: this.form.field('platform'),
      kernelBackend: this.form.field('kernelBackend'),
      strategy: this.form.field('strategy'),
    }
  }

  inject(): ConfigToolCardFace {
    return { hooks: { configToolCard: this.store }, ...this.form.actions() }
  }
}
