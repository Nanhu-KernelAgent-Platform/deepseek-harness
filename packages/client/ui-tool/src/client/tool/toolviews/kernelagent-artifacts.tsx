/** Structured source-file presentation for KernelAgent results. */

import { CodeBlock, MarkdownText } from '@deepseek-ai/dsh-client-ui-primitives'
import type { TranslateNS } from '@deepseek-ai/dsh-client-ui-slots'
import css from './kernelagent-artifacts.module.css'

export interface KernelArtifact {
  fileName: string
  source: string
}

/** Extract the named code fences emitted by kernelagent-tool. */
export function parseKernelArtifacts(report: string): KernelArtifact[] {
  const files: KernelArtifact[] = []
  const pattern = /━━━ ([^\n]+) ━━━\s*\n+```([\w-]+):([^\n]+)\n([\s\S]*?)\n```/g
  for (const match of report.matchAll(pattern)) {
    files.push({
      fileName: match[3] || match[1] || 'kernel.txt',
      source: match[4] || '',
    })
  }
  return files
}

export function KernelAgentArtifacts({ report = '', chart = '', files: sourceFiles, t }: {
  report?: string
  chart?: string
  files?: KernelArtifact[]
  t: TranslateNS<'conversation'>
}) {
  const files = sourceFiles ?? parseKernelArtifacts(report)
  const chartText = chart === '' ? '' : '\n\n```perfchart\n' + chart + '\n```'
  if (files.length === 0) {
    return (
      <MarkdownText
        text={report + chartText}
        codeLabels={{ copyLabel: t('copy'), copiedLabel: t('copied'), downloadLabel: '下载' }}
      />
    )
  }

  return (
    <div className={css.fileList}>
      {files.map(file => (
        <section className={css.fileCard} key={file.fileName}>
          <div className={css.fileName}>{file.fileName}</div>
          <CodeBlock
            code={file.source}
            lang={file.fileName.endsWith('.py') ? 'python' : /\.(mu|cu|cpp|cc|cxx)$/.test(file.fileName) ? 'cpp' : 'text'}
            fileName={file.fileName}
            copyLabel={t('copy')}
            copiedLabel={t('copied')}
            downloadLabel="下载"
          />
        </section>
      ))}
      {chartText !== '' && <MarkdownText text={chartText} />}
    </div>
  )
}
