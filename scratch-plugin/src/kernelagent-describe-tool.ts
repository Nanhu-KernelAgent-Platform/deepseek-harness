import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import {
  CONFIG_TOOL_SETTINGS_NAMESPACE,
  resolveApiKey,
  type Config,
} from './kernelagent-config-tool.ts'
import {
  buildKernelAgentPrompt,
  setPreparedPrompt,
  type DescribeKind,
} from './kernelagent-prompt-store.ts'

const DESCRIBE2_SYSTEM_FORWARD = `You turn Describe1 operator formulas into Describe2: one PyTorch nn.Module whose forward body uses those formulas.

Rules:
- Output ONLY Python source. No Markdown fences. No prose before or after the class.
- Exactly one class subclassing torch.nn.Module (name it Model unless Describe1 names another).
- Implement __init__ (create parameters/buffers referenced by the formulas) and forward.
- The forward method MUST contain the Describe1 formula lines (or their direct equivalent), not a rewritten algorithm that drops them.
- Define top-level get_inputs() returning a list or tuple of representative CPU tensors and scalar forward arguments.
- Define top-level get_init_inputs() returning constructor arguments, or an empty list when Model takes none.
- Keep imports minimal: torch and torch.nn as needed.

Example:
Describe1 formula:
    output = torch.matmul(x, self.weight)

Describe2 must look like:
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(x, self.weight)
        return output

def get_inputs():
    return [torch.randn(32, 64)]

def get_init_inputs():
    return [64, 128]`

const DESCRIBE2_SYSTEM_FORWARD_BACKWARD = `You turn Describe1 operator formulas into Describe2 for one bound forward/backward operator.

Rules:
- Output ONLY Python source. No Markdown fences. No prose before or after the code.
- Provide one torch.autograd.Function subclass whose forward implements the Describe1 formula and whose backward analytically derives its gradients.
- Provide one torch.nn.Module wrapper named Model that calls that Function, so forward and backward are tested and generated as a single operator.
- Save every value required by backward in ctx and return gradients for every differentiable forward input or parameter.
- The KernelAgent task must generate, verify, and bind forward and backward in one native kernel bundle, then optimize their performance separately.
- Do not create a second independent backward task.
- Define top-level get_inputs() returning a list or tuple of representative CPU tensors and scalar forward arguments.
- Define top-level get_init_inputs() returning constructor arguments, or an empty list when Model takes none.
- Keep imports minimal: torch and torch.nn as needed.

Example:
Describe1 formula:
    output = torch.matmul(x, self.weight)

Describe2 must look like:
import torch

class ModelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        output = torch.matmul(x, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_x = torch.matmul(grad_output, weight.transpose(-2, -1))
        grad_weight = torch.matmul(x.transpose(-2, -1), grad_output)
        return grad_x, grad_weight

class Model(torch.nn.Module):
    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return ModelFunction.apply(x, weight)

def get_inputs():
    return [torch.randn(32, 64), torch.randn(64, 128)]

def get_init_inputs():
    return []`

const BACKWARD_INTENT = /反向|backward|reverse\s*op|reverse\s*operator|gradient|grad_output|反传/i

/**
 * Resolve whether Describe2 should cover forward only or bound forward/backward.
 * Explicit kind wins; otherwise detect reverse intent in Describe1 text.
 * @param kind - optional tool argument.
 * @param describe1 - dialog text.
 * @returns forward or forward_backward.
 */
export function resolveDescribeKind(kind: unknown, describe1: string): DescribeKind {
  if (kind === 'forward' || kind === 'forward_backward') return kind
  return BACKWARD_INTENT.test(describe1) ? 'forward_backward' : 'forward'
}

/**
 * Build the user message that asks the LLM for Describe2.
 * @param describe1 - dialog / formula text.
 * @param kind - forward only or bound forward/backward.
 * @returns chat user content.
 */
export function buildDescribe2UserMessage(
  describe1: string,
  kind: DescribeKind = 'forward',
): string {
  if (kind === 'forward_backward') {
    return [
      'Convert the following Describe1 into one bound forward/backward Describe2.',
      'Derive the backward formula from the forward formula.',
      'Describe2 must contain a torch.autograd.Function with both forward and backward plus an nn.Module wrapper named Model.',
      'KernelAgent must generate, verify, and bind both directions as one native kernel bundle, then optimize each direction separately.',
      '',
      'Describe1:',
      describe1.trim(),
    ].join('\n')
  }
  return [
    'Convert the following Describe1 into Describe2.',
    'Describe2 must be a Class with a forward method; forward must include the Describe1 formula.',
    '',
    'Describe1:',
    describe1.trim(),
  ].join('\n')
}

/**
 * Flatten model-visible message content blocks into plain text.
 * @param content - content blocks from a derived message.
 * @returns concatenated text.
 */
export function contentBlocksToText(content: ReadonlyArray<{ type: string; text?: string }>): string {
  return content
    .map(block => (block.type === 'text' && typeof block.text === 'string' ? block.text : ''))
    .join('')
    .trim()
}

/**
 * Collect dialog text used as Describe1 from derived session messages.
 * Defaults to every user message; optional `dialog` overrides that source.
 * @param messages - derived session messages.
 * @param dialogOverride - optional explicit dialog text from the tool call.
 * @returns Describe1 text.
 */
export function collectDescribe1(
  messages: ReadonlyArray<{ role: string; content: ReadonlyArray<{ type: string; text?: string }> }>,
  dialogOverride?: string,
): string {
  const override = dialogOverride?.trim()
  if (override) return override
  const parts = messages
    .filter(message => message.role === 'user')
    .map(message => contentBlocksToText(message.content))
    .filter(text => text.length > 0)
  return parts.join('\n\n').trim()
}

/**
 * Strip accidental Markdown fences from a model reply.
 * @param text - raw model content.
 * @returns Python source without outer fences.
 */
export function stripCodeFences(text: string): string {
  const trimmed = text.trim()
  const fenced = trimmed.match(/^```(?:python|py)?\s*\n([\s\S]*?)\n```$/i)
  return fenced ? fenced[1].trim() : trimmed
}

/**
 * Normalize a KernelAgent settings baseURL into a chat-completions POST URL.
 * Settings often store an OpenAI-compatible root (what the Python bridge wants);
 * Describe2 must POST to `.../chat/completions`.
 * @param baseURL - value from KernelAgent settings.
 * @returns absolute chat-completions endpoint.
 */
export function resolveChatCompletionsUrl(baseURL: string): string {
  const trimmed = baseURL.trim().replace(/\/+$/, '')
  if (trimmed.endsWith('/chat/completions')) return trimmed
  if (trimmed === 'https://zapi.deuo.top') return 'https://zapi.deuo.top/v1/chat/completions'
  return `${trimmed}/chat/completions`
}

/**
 * Call an OpenAI-compatible chat endpoint to produce Describe2.
 * @param args - request fields.
 * @returns model text.
 */
export async function generateDescribe2(args: {
  baseURL: string
  apiKey: string
  model: string
  describe1: string
  kind: DescribeKind
  signal?: AbortSignal
}): Promise<string> {
  const endpoint = resolveChatCompletionsUrl(args.baseURL)
  const system = args.kind === 'forward_backward' ? DESCRIBE2_SYSTEM_FORWARD_BACKWARD : DESCRIBE2_SYSTEM_FORWARD
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${args.apiKey}`,
    },
    body: JSON.stringify({
      model: args.model,
      messages: [
        { role: 'system', content: system },
        {
          role: 'user',
          content: buildDescribe2UserMessage(args.describe1, args.kind),
        },
      ],
    }),
    signal: args.signal,
  })
  const raw = await response.text()
  if (!response.ok) {
    throw new Error(`Describe2 LLM error (${response.status}) at ${endpoint}: ${raw.slice(0, 500)}`)
  }
  const contentType = response.headers.get('content-type') ?? ''
  if (!contentType.includes('json') && /^\s*</.test(raw)) {
    throw new Error(
      `Describe2 expected JSON from ${endpoint}, but got HTML. Check KernelAgent settings baseURL/apiKey (got content-type=${contentType || 'missing'}).`,
    )
  }
  let data: { choices?: Array<{ message?: { content?: string } }> }
  try {
    data = JSON.parse(raw) as { choices?: Array<{ message?: { content?: string } }> }
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)
    throw new Error(
      `Describe2 JSON parse failed at ${endpoint}: ${detail}. Body starts with: ${raw.slice(0, 120)}`,
    )
  }
  const content = data.choices?.[0]?.message?.content
  if (typeof content !== 'string' || content.trim() === '') {
    throw new Error('Describe2 LLM returned empty content')
  }
  return stripCodeFences(content)
}

export const name = 'kernelagent-describe-tool'
export const inject = ['tools', 'systemPrompt', 'settings']

export function apply(ctx: Context) {
  ctx.systemPrompt.section({
    name: 'tool:kernelagent-describe',
    order: 205,
    text: `Before calling kernelagent in generate, fuse, or optimize mode, you MUST call kernelagent_describe first.
kernelagent_describe collects the dialog as Describe1, generates Describe2, stores the combined prompt, and returns it.
Use kind=forward (default) for a PyTorch nn.Module whose forward contains the formula.
Use kind=forward_backward when the user wants gradients or a reverse/backward operator. It derives Describe2 for both directions so one following kernelagent call generates, binds, and verifies the complete operator, then automatically optimizes forward and backward separately. Never split this into separate forward and backward kernelagent calls.
Do not invent problem_code for gated modes: the gate injects the prepared prompt.
run_example mode does not require kernelagent_describe.`,
  })

  ctx.tools.register(defineTool({
    name: 'kernelagent_describe',
    description: 'Prepare one KernelAgent prompt from the dialog. Generate either a forward-only PyTorch class or a bound forward/backward reference whose gradients are derived from the forward formula.',
    parameters: {
      dialog: {
        type: 'string',
        description: 'Optional dialog override. Omit to use all user messages in this session as Describe1.',
      },
      kind: {
        type: 'string',
        enum: ['forward', 'forward_backward'],
        description: 'forward: nn.Module with forward formula. forward_backward: one bound reference with derived backward. Omit to infer from dialog.',
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          kind: { type: 'string', enum: ['forward', 'forward_backward'] },
          describe1: { type: 'string' },
          describe2: { type: 'string' },
          prompt: { type: 'string' },
          ready: { type: 'boolean' },
        },
      },
      render: (_args, value) => [{
        type: 'text',
        text: [
          `KernelAgent describe prepared (${value.kind}).`,
          `Describe1 chars: ${value.describe1.length}`,
          `Describe2 chars: ${value.describe2.length}`,
          'Call kernelagent next; problem_code will be taken from this prepared prompt.',
        ].join('\n'),
      }],
    },
    async execute(args, exec) {
      const agent = exec.agent
      if (!agent) throw new Error('kernelagent_describe requires a live agent')

      const globalConfig = ctx.settings.get(CONFIG_TOOL_SETTINGS_NAMESPACE) as Config | undefined
      if (!globalConfig) throw new Error('KernelAgent settings are not registered')
      const apiKey = await resolveApiKey(ctx, globalConfig)
      if (!apiKey) throw new Error('No API key available for Describe2 generation')

      const describe1 = collectDescribe1(agent.session.deriveMessages(), args.dialog)
      if (!describe1) {
        throw new Error('Describe1 is empty: provide dialog text or send a user message first')
      }

      const kind = resolveDescribeKind(args.kind, describe1)
      const sessionId = String(agent.session.id)

      const describe2 = await generateDescribe2({
        baseURL: globalConfig.baseURL,
        apiKey,
        model: globalConfig.modelName,
        describe1,
        kind,
        signal: exec.signal,
      })
      const prompt = buildKernelAgentPrompt(describe1, describe2, kind)
      setPreparedPrompt(sessionId, {
        describe1,
        describe2,
        prompt,
        kind,
        preparedAt: Date.now(),
      })

      return { kind, describe1, describe2, prompt, ready: true as const }
    },
  }))

  console.log('[kernelagent-describe-tool] registered')
}
