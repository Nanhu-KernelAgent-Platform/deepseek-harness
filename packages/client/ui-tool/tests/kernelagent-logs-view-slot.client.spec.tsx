// @vitest-environment jsdom
// Registration probe: ui-tool's KernelAgent logs tab lands on the
// conversation.view ring that ui-conversation declares. Fake connection /
// remote cover the Tool plugin's inject list; no outlet render is needed.

import { describe, expect, it, vi } from 'vitest'
import { SlotTestRuntime, usePinnedBrowserLanguages, stubSettingsScope } from '@deepseek-ai/dsh-client-test-runtime'
import { resolveSlotLabel } from '@deepseek-ai/dsh-client-ui-slots'
import { LocaleRuntime } from '@deepseek-ai/dsh-client-locale/client'
import type { SessionId } from '@deepseek-ai/dsh-client-runtime/client'
import { apply as applyConversation, inject as injectConversation } from '@deepseek-ai/dsh-client-ui-conversation/client'
import { apply as applyTool, inject as injectTool } from '../src/client/apply.ts'

// The conversation service reads its initial locale from the browser; this
// spec asserts the shipped Chinese tab label, so it states the browser it
// assumes.
usePinnedBrowserLanguages('zh-CN')

const SID = 's1' as SessionId

async function bench() {
  const runtime = await SlotTestRuntime.create()
  runtime.provide('connection', {
    api: { settings: {} },
    isLoopback: false,
    hostDescription: { getSnapshot: () => undefined, subscribe: () => () => {} },
  })
  runtime.provide('remote', { $on: () => () => {} })
  runtime.provide('settingsScope', { bind: () => stubSettingsScope().scope } as never)
  runtime.provide('layout', { openDetails: vi.fn(), closeDetails: vi.fn() })
  const locale = new LocaleRuntime(runtime.ctx)
  runtime.provide('locale', locale)
  runtime.slots.installLocale(locale)
  await runtime.sessions.add({ id: SID, summary: { title: 'S', displayTitle: 'S' } }, { current: false })

  // Declared by ui-layout's root entry in production; the test root declares
  // them here so conversation contributions land.
  await runtime.root.declare({
    conversation: { kind: 'single', scope: 'session-maybe' },
    details: { kind: 'single', scope: 'session' },
    'settings.general.item': { kind: 'list', scope: 'root' },
  }, (_p: { renderSlot?: unknown }) => null)

  await runtime.mount({ inject: [...injectConversation], apply: applyConversation })
  const tool = await runtime.mount({ inject: [...injectTool], apply: applyTool })
  return { runtime, tool, slots: runtime.slots }
}

describe('kernelagent-logs conversation.view registration', () => {
  it("registers id 'kernelagent-logs' with label 日志 on conversation.view", async () => {
    const b = await bench()
    const entry = b.slots.entries('conversation.view').find(e => e.options.id === 'kernelagent-logs')
    expect(entry).toBeDefined()
    expect(resolveSlotLabel(entry?.options.label)).toBe('日志')
    expect(entry?.options.order).toBe(20)
    await b.tool.dispose()
    expect(b.slots.entries('conversation.view').some(e => e.options.id === 'kernelagent-logs')).toBe(false)
    await b.runtime.dispose()
  })
})
