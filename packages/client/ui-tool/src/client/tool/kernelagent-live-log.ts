/**
 * Process-local KernelAgent live-log buffer for the Web tool card and the
 * conversation "日志" view tab. Hosted as a HostObservable so the slot renderer
 * binds `useKernelAgentLiveLogs` without components owning subscription
 * machinery. Entries survive settle until the page unloads; they are never
 * written to the session log.
 */

import type { HostObservable } from '@deepseek-ai/dsh-client-ui-slots'
import type { KernelAgentLogUpdate } from '@deepseek-ai/dsh-api-remotes/client'

/** One call's retained log text plus the session that owns it. */
export interface KernelAgentLiveLogEntry {
  sessionId: string
  callId: string
  text: string
}

/** Snapshot keyed by callId, with per-session latest-call pointers. */
export interface KernelAgentLiveLogMap {
  readonly byCallId: Readonly<Record<string, KernelAgentLiveLogEntry>>
  /** sessionId → most recently updated callId in that session. */
  readonly latestBySession: Readonly<Record<string, string>>
}

const EMPTY: KernelAgentLiveLogMap = Object.freeze({
  byCallId: Object.freeze({}),
  latestBySession: Object.freeze({}),
})

/** Mutable live-log source plus the remote-event ingest callback. */
export interface KernelAgentLiveLogStore extends HostObservable<KernelAgentLiveLogMap> {
  /** Apply one whole-buffer update from `kernelagent/log`. */
  ingest(update: KernelAgentLogUpdate): void
}

/**
 * Create an empty live-log store. Snapshot identity is stable between changes.
 * @returns a store the Tool plugin registers into inject.hooks and feeds from remote.$on.
 */
export function createKernelAgentLiveLogStore(): KernelAgentLiveLogStore {
  let snapshot: KernelAgentLiveLogMap = EMPTY
  const listeners = new Set<() => void>()

  const notify = (): void => {
    for (const listener of listeners) listener()
  }

  return {
    getSnapshot: () => snapshot,
    subscribe: (fn) => {
      listeners.add(fn)
      return () => { listeners.delete(fn) }
    },
    ingest(update) {
      const prev = snapshot.byCallId[update.callId]
      if (prev !== undefined
        && prev.text === update.text
        && prev.sessionId === update.sessionId) {
        return
      }
      snapshot = Object.freeze({
        byCallId: Object.freeze({
          ...snapshot.byCallId,
          [update.callId]: Object.freeze({
            sessionId: update.sessionId,
            callId: update.callId,
            text: update.text,
          }),
        }),
        latestBySession: Object.freeze({
          ...snapshot.latestBySession,
          [update.sessionId]: update.callId,
        }),
      })
      notify()
    },
  }
}
