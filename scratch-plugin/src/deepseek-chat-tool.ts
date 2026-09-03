import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// 读取密钥
const secretsPath = resolve(__dirname, 'secrets.json')
const secrets = JSON.parse(readFileSync(secretsPath, 'utf-8'))
const apiKey: string = secrets.deepseekApiKey

if (!apiKey) {
  throw new Error('[deepseek-chat-tool] secrets.json 中缺少 deepseekApiKey')
}
console.log('[deepseek-chat-tool] 已加载，API Key 前缀: ' + apiKey.substring(0, 10) + '...')

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
