import { useEffect, useRef, useState, useCallback } from 'react'
import css from './PerfChartBlock.module.css'

export interface PerfChartBlockProps {
  /** JSON string containing chart data. */
  dataJson: string
}

interface PerfChartData {
  title?: string
  unit?: string
  data: Array<{
    label: string
    value: number
    color?: string
  }>
  improvement_pct?: number | null
}

const DEFAULT_COLORS = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272']

export function PerfChartBlock({ dataJson }: PerfChartBlockProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [error, setError] = useState<string | null>(null)

  const drawChart = useCallback((canvas: HTMLCanvasElement, chartData: PerfChartData) => {
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const cssWidth = 640
    const cssHeight = 360
    canvas.width = cssWidth * dpr
    canvas.height = cssHeight * dpr
    canvas.style.width = `${cssWidth}px`
    canvas.style.height = `${cssHeight}px`
    ctx.scale(dpr, dpr)

    // Clear
    ctx.clearRect(0, 0, cssWidth, cssHeight)

    // Layout
    const padding = { top: 50, right: 30, bottom: 60, left: 70 }
    const chartW = cssWidth - padding.left - padding.right
    const chartH = cssHeight - padding.top - padding.bottom

    const items = chartData.data ?? []
    if (items.length === 0) {
      ctx.fillStyle = '#999'
      ctx.font = '14px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('No performance data available', cssWidth / 2, cssHeight / 2)
      return
    }

    const maxValue = Math.max(...items.map(d => d.value), 0.0001)
    const barGap = Math.min(40, chartW / (items.length * 4))
    const barWidth = (chartW - barGap * (items.length + 1)) / items.length

    // Title
    const title = chartData.title ?? 'Performance Comparison'
    ctx.fillStyle = '#333'
    ctx.font = 'bold 16px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(title, cssWidth / 2, 24)

    // Y-axis grid lines
    const gridCount = 5
    ctx.strokeStyle = '#e0e0e0'
    ctx.lineWidth = 1
    ctx.font = '12px sans-serif'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    ctx.fillStyle = '#666'

    for (let i = 0; i <= gridCount; i++) {
      const y = padding.top + chartH - (i / gridCount) * chartH
      const val = (maxValue * i) / gridCount

      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(padding.left + chartW, y)
      ctx.stroke()

      const unit = chartData.unit ?? 'ms'
      ctx.fillText(`${val.toFixed(4)} ${unit}`, padding.left - 10, y)
    }

    // Y-axis label
    ctx.save()
    ctx.translate(16, cssHeight / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillStyle = '#666'
    ctx.font = '13px sans-serif'
    ctx.fillText(`Latency (${chartData.unit ?? 'ms'})`, 0, 0)
    ctx.restore()

    // Bars
    items.forEach((item, index) => {
      const color = (item.color ?? DEFAULT_COLORS[index % DEFAULT_COLORS.length]) as string
      const barH = (item.value / maxValue) * chartH
      const x = padding.left + barGap + index * (barWidth + barGap)
      const y = padding.top + chartH - barH

      // Bar shadow
      ctx.fillStyle = 'rgba(0,0,0,0.08)'
      ctx.fillRect(x + 2, y + 2, barWidth, barH)

      // Bar gradient
      const grad = ctx.createLinearGradient(0, y + barH, 0, y)
      grad.addColorStop(0, color)
      grad.addColorStop(1, lightenColor(color, 0.25))
      ctx.fillStyle = grad
      ctx.fillRect(x, y, barWidth, barH)

      // Value label on top
      ctx.fillStyle = '#333'
      ctx.font = 'bold 12px sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'bottom'
      const unit = chartData.unit ?? 'ms'
      ctx.fillText(`${item.value.toFixed(5)} ${unit}`, x + barWidth / 2, y - 6)

      // X-axis label
      ctx.fillStyle = '#555'
      ctx.font = '13px sans-serif'
      ctx.textBaseline = 'top'
      ctx.fillText(item.label, x + barWidth / 2, padding.top + chartH + 10)
    })

    // Improvement badge
    if (chartData.improvement_pct !== undefined && chartData.improvement_pct !== null) {
      const pct = Number(chartData.improvement_pct)
      const badgeText = `Improvement: ${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`
      ctx.font = 'bold 13px sans-serif'
      const textW = ctx.measureText(badgeText).width
      const bx = cssWidth - padding.right - textW - 16
      const by = 8
      const bw = textW + 16
      const bh = 24

      ctx.fillStyle = pct > 0 ? '#e6f7e6' : '#f0f0f0'
      ctx.strokeStyle = pct > 0 ? '#52c41a' : '#999'
      ctx.lineWidth = 1
      roundRect(ctx, bx, by, bw, bh, 4)
      ctx.fill()
      ctx.stroke()

      ctx.fillStyle = pct > 0 ? '#389e0d' : '#666'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(badgeText, bx + bw / 2, by + bh / 2 + 1)
    }

    // Border
    ctx.strokeStyle = '#ddd'
    ctx.lineWidth = 1
    ctx.strokeRect(padding.left, padding.top, chartW, chartH)
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    let chartData: PerfChartData
    try {
      chartData = JSON.parse(dataJson) as PerfChartData
      if (!Array.isArray(chartData.data)) {
        throw new Error('"data" field must be an array')
      }
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : String(e)
      setError(`Invalid chart data: ${message}`)
      return
    }

    setError(null)
    drawChart(canvas, chartData)
  }, [dataJson, drawChart])

  if (error) {
    return (
      <div className={css.error}>
        <span className={css.errorIcon}>⚠️</span>
        <span>{error}</span>
      </div>
    )
  }

  return (
    <div className={css.container}>
      <canvas ref={canvasRef} className={css.canvas} />
    </div>
  )
}

function lightenColor(hex: string, amount: number): string {
  const num = parseInt(hex.slice(1), 16)
  const r = Math.min(255, ((num >> 16) & 0xff) + Math.round(255 * amount))
  const g = Math.min(255, ((num >> 8) & 0xff) + Math.round(255 * amount))
  const b = Math.min(255, (num & 0xff) + Math.round(255 * amount))
  return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.lineTo(x + w - r, y)
  ctx.quadraticCurveTo(x + w, y, x + w, y + r)
  ctx.lineTo(x + w, y + h - r)
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h)
  ctx.lineTo(x + r, y + h)
  ctx.quadraticCurveTo(x, y + h, x, y + h - r)
  ctx.lineTo(x, y + r)
  ctx.quadraticCurveTo(x, y, x + r, y)
  ctx.closePath()
}
