import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
const apiKey = process.env.DEEPSEEK_API_KEY

if (!apiKey) {
  throw new Error('[deepseek-chat-tool] Missing required environment variable: DEEPSEEK_API_KEY')
}

export const name = 'deepseek-chat-tool'
export const inject = ['tools', 'systemPrompt']

export function apply(ctx: Context) {
  // 关键：告诉模型这个工具存在
  ctx.systemPrompt.section({
    name: 'tool:deepseek-chat',
    order: 200,
    text: 'Use the deepseek-chat tool to converse with DeepSeek AI model when you need external AI assistance.',
  })

  ctx.tools.register(defineTool({
    name: 'deepseek-chat',
    description: 'Send a message to the DeepSeek AI model and return its response.',
    parameters: {
      message: {
        type: 'string',
        required: true,
        description: 'The message to send to the AI.',
      },
      model: {
        type: 'string',
        description: 'Model ID to use. Defaults to deepseek-chat.',
        default: 'deepseek-chat',
      },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args, exec) {
      console.log('[deepseek-chat] 正在调用 API...')

      const response = await fetch('https://api.deepseek.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: args.model || 'deepseek-chat',
          messages: [{ role: 'user', content: args.message }],
        }),
        signal: exec.signal,
      })

      if (!response.ok) {
        const err = await response.text()
        throw new Error(`DeepSeek API error (${response.status}): ${err}`)
      }

      const data = await response.json() as any
      return data.choices[0].message.content
    },
  }))

  console.log('[deepseek-chat-tool] 已注册到 ctx.tools')
}
