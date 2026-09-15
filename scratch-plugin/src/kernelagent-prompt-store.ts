/**
 * Session-scoped store for KernelAgent prompts prepared by the describe
 * middleware. `kernelagent` may run generate/fuse/optimize only after a
 * prepared prompt exists for the calling session.
 */

/** Modes that require a prompt prepared by kernelagent_describe. */
const PROMPT_GATED_MODES = new Set(['generate', 'fuse', 'optimize'])

/** Which operator side Describe2 targets. */
export type DescribeKind = 'forward' | 'forward_backward'

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
  /** PyTorch source generated as Describe2. */
  describe2: string
  /** Combined prompt passed to KernelAgent as problem_code. */
  prompt: string
  /** Whether Describe2 covers forward only or bound forward/backward behavior. */
  kind: DescribeKind
  /** Wall-clock time when the prompt was prepared. */
  preparedAt: number
}

const store = new Map<string, PreparedKernelAgentPrompt>()

/**
 * Build the KernelAgent problem prompt from Describe1 and Describe2.
 * @param describe1 - dialog text.
 * @param describe2 - generated PyTorch class.
 * @param kind - forward-only or bound forward/backward Describe2.
 * @returns the combined prompt string.
 */
export function buildKernelAgentPrompt(
  describe1: string,
  describe2: string,
  kind: DescribeKind = 'forward',
): string {
  const describe2Heading = kind === 'forward_backward'
    ? '## Describe2 (one bound operator implementation with forward and derived backward)'
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
 * @param sessionId - owning session id.
 * @param prepared - prompt payload.
 */
export function setPreparedPrompt(sessionId: string, prepared: PreparedKernelAgentPrompt): void {
  store.set(sessionId, prepared)
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
}
