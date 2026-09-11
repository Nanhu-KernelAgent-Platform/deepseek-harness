/**
 * Session-scoped store for KernelAgent prompts prepared by the describe
 * middleware. `kernelagent` may run generate/fuse/optimize only after a
 * prepared prompt exists for the calling session.
 */

/** Modes that require a prompt prepared by kernelagent_describe. */
const PROMPT_GATED_MODES = new Set(['generate', 'fuse', 'optimize'])

/** Which operator side Describe2 targets. */
export type DescribeKind = 'forward' | 'backward'

/**
 * Whether a kernelagent call must consume a prepared describe prompt.
 * @param mode - tool mode argument.
 * @returns true when generate/fuse/optimize.
 */
export function requiresPreparedPrompt(mode: unknown): boolean {
  return typeof mode === 'string' && PROMPT_GATED_MODES.has(mode)
}

export interface PreparedKernelAgentPrompt {
  /** Dialog text collected as Describe1. */
  describe1: string
  /** PyTorch class generated as Describe2 (forward or backward). */
  describe2: string
  /** Combined prompt passed to KernelAgent as problem_code. */
  prompt: string
  /** Whether Describe2 is the forward or backward operator class. */
  kind: DescribeKind
  /** Wall-clock time when the prompt was prepared. */
  preparedAt: number
}

const store = new Map<string, PreparedKernelAgentPrompt>()
/** Survives prompt consumption so a later backward describe can reuse it. */
const lastForwardDescribe2 = new Map<string, string>()

/**
 * Build the KernelAgent problem prompt from Describe1 and Describe2.
 * @param describe1 - dialog text.
 * @param describe2 - generated PyTorch class.
 * @param kind - forward or backward Describe2.
 * @returns the combined prompt string.
 */
export function buildKernelAgentPrompt(
  describe1: string,
  describe2: string,
  kind: DescribeKind = 'forward',
): string {
  const describe2Heading = kind === 'backward'
    ? '## Describe2 (class that implements backward for the operator)'
    : '## Describe2 (nn.Module class whose forward contains the Describe1 formula)'
  return [
    '## Describe1 (operator formula / dialog)',
    describe1.trim(),
    '',
    describe2Heading,
    describe2.trim(),
  ].join('\n')
}

/**
 * Store a prepared prompt for a session, replacing any previous one.
 * Forward Describe2 is also retained for a later backward describe.
 * @param sessionId - owning session id.
 * @param prepared - prompt payload.
 */
export function setPreparedPrompt(sessionId: string, prepared: PreparedKernelAgentPrompt): void {
  store.set(sessionId, prepared)
  if (prepared.kind === 'forward') {
    lastForwardDescribe2.set(sessionId, prepared.describe2)
  }
}

/**
 * Read the last successful forward Describe2 for a session.
 * @param sessionId - owning session id.
 * @returns forward class source, or undefined when absent.
 */
export function getLastForwardDescribe2(sessionId: string): string | undefined {
  return lastForwardDescribe2.get(sessionId)
}

/**
 * Read the prepared prompt without consuming it.
 * @param sessionId - owning session id.
 * @returns the prepared payload, or undefined when absent.
 */
export function peekPreparedPrompt(sessionId: string): PreparedKernelAgentPrompt | undefined {
  return store.get(sessionId)
}

/**
 * Take and remove the prepared prompt for a session.
 * Does not clear the retained forward Describe2.
 * @param sessionId - owning session id.
 * @returns the prepared payload, or undefined when absent.
 */
export function takePreparedPrompt(sessionId: string): PreparedKernelAgentPrompt | undefined {
  const prepared = store.get(sessionId)
  if (prepared === undefined) return undefined
  store.delete(sessionId)
  return prepared
}

/**
 * Drop any prepared prompt for a session.
 * @param sessionId - owning session id.
 */
export function clearPreparedPrompt(sessionId: string): void {
  store.delete(sessionId)
}

/** Test-only: empty the store. */
export function resetPreparedPromptStore(): void {
  store.clear()
  lastForwardDescribe2.clear()
}
