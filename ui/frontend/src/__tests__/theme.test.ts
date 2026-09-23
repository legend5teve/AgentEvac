/// <reference types="vite/client" />
import { describe, expect, it } from 'vitest'

// A Tailwind class naming a colour the theme does not define emits no CSS at all. Nothing
// warns, the build succeeds, and the element renders without that colour, which is how the
// Author view shipped with a selected alert area that looked exactly like an unselected
// one. `border-accent`, `bg-surface-sunken` and `text-status-good` were all invented names,
// and `status.good` in particular is one word away from the real `status.nominal`. These
// tests read the class names back out of the source and hold them to the palette
// `tailwind.config.js` actually defines.

/** Colour tokens the theme defines, as the utility suffixes they produce. */
const THEME_COLORS = new Set([
  'ink-bg', 'ink-panel', 'ink-raised', 'ink-line', 'ink-text', 'ink-muted', 'ink-faint',
  'status-nominal', 'status-caution', 'status-hazard', 'status-moving', 'status-idle',
  'transparent', 'current', 'inherit', 'white', 'black',
])

/** Utilities that share a prefix with a colour utility and name something else. */
const NOT_A_COLOUR = new Set([
  'dashed', 'solid', 'dotted', 'double', 'hidden', 'none',
  'left', 'center', 'right', 'justify', 'start', 'end',
  'micro', 'small', 'base', 'panel', 'view', 'readout',
  'wide', 'wider', 'widest', 'tight', 'tighter', 'normal',
  'collapse', 'separate', 'clip', 'ellipsis', 'nowrap', 'wrap', 'balance', 'pretty',
  'top', 'bottom', 'middle', 'baseline',
  'uppercase', 'lowercase', 'capitalize',
  'thin', 'light', 'medium', 'semibold', 'bold', 'extrabold', 'italic',
  'opacity', 'auto', 'fixed', 'local', 'scroll', 'cover', 'contain', 'repeat',
  // Side and axis suffixes, as in border-t and divide-y.
  't', 'b', 'l', 'r', 'x', 'y', 's', 'e',
])

const COLOUR_PREFIX = /\b(border|bg|text|ring|outline|divide|placeholder|from|via|to)-([a-z][a-z0-9-]*)/g

/**
 * Class names as written in the source.
 *
 * Only the contents of `className` are read, because a MapLibre layer id and a style
 * property look identical to a colour utility otherwise. `area-outline-ordered` is a layer
 * id and `text-field` is a symbol layout property, and neither is CSS.
 */
function classNames(source: string): string[] {
  const out: string[] = []
  for (const match of source.matchAll(/className=(?:"([^"]*)"|\{`([^`]*)`\}|\{'([^']*)'\})/g)) {
    out.push(match[1] ?? match[2] ?? match[3] ?? '')
  }
  return out
}

export function undefinedColours(source: string): string[] {
  const found: string[] = []
  for (const value of classNames(source)) {
    for (const [token, , name] of value.matchAll(COLOUR_PREFIX)) {
      if (name.startsWith('[')) continue // arbitrary value, resolved by Tailwind directly
      if (/^\d/.test(name)) continue // a width or a size, as in border-2
      const root = name.split('-')[0]
      if (NOT_A_COLOUR.has(name) || NOT_A_COLOUR.has(root)) continue
      if (THEME_COLORS.has(name)) continue
      found.push(token)
    }
  }
  return found
}

// Raw source for every module under src, which needs no node types to read.
const SOURCES = import.meta.glob('../**/*.{ts,tsx}', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

const FILES = Object.entries(SOURCES).filter(([file]) => !file.includes('__tests__'))

describe('theme colours', () => {
  it('finds the source to check', () => {
    expect(FILES.length).toBeGreaterThan(10)
  })

  it.each(FILES)('%s names only colours the theme defines', (_file, source) => {
    expect(undefinedColours(source)).toEqual([])
  })

  // The guard is only worth having if it would have caught the bug it was written for.
  it('catches the invented names that caused this', () => {
    expect(
      undefinedColours('<div className="rounded border border-accent bg-surface-sunken" />'),
    ).toEqual(['border-accent', 'bg-surface-sunken'])
    expect(undefinedColours('<p className="text-status-good">ok</p>')).toEqual([
      'text-status-good',
    ])
  })

  it('does not flag a utility that only looks like a colour', () => {
    expect(
      undefinedColours('<p className="border-t border-dashed text-small text-left uppercase" />'),
    ).toEqual([])
  })

  it('accepts the tokens the theme really defines', () => {
    expect(
      undefinedColours('<p className="border-ink-line bg-ink-bg text-status-nominal" />'),
    ).toEqual([])
  })
})
