// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { SelectField } from '../src/client/fields.tsx'

afterEach(cleanup)

describe('KernelAgent SelectField', () => {
  it('only offers supported values and stages the selected wire value', () => {
    const onEdit = vi.fn()
    render(
      <SelectField
        id="platform"
        label="Platform"
        hint="Target platform"
        text="musa"
        overridden={false}
        invalid={false}
        overriddenLabel="Overridden"
        resetLabel="Reset"
        invalidLabel="Invalid"
        disabled={false}
        options={['cuda', 'musa', 'xpu']}
        onEdit={onEdit}
        onReset={vi.fn()}
      />,
    )

    const select = screen.getByLabelText('Platform')
    expect(Array.from(select.querySelectorAll('option')).map(option => option.value))
      .toEqual(['cuda', 'musa', 'xpu'])
    fireEvent.change(select, { target: { value: 'xpu' } })
    expect(onEdit).toHaveBeenCalledWith('xpu')
  })

  it('keeps a legacy invalid value visible so it can be replaced', () => {
    render(
      <SelectField
        id="strategy"
        label="Strategy"
        hint="Search strategy"
        text="greddy"
        overridden
        invalid
        overriddenLabel="Overridden"
        resetLabel="Reset"
        invalidLabel="Invalid"
        disabled={false}
        options={['beam_search', 'greedy']}
        onEdit={vi.fn()}
        onReset={vi.fn()}
      />,
    )

    expect(screen.getByLabelText('Strategy')).toHaveProperty('value', 'greddy')
    expect(screen.getByText('Invalid')).toBeTruthy()
  })
})
