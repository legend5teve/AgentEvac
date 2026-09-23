import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, ApiError } from '../state/api'
import { integer, simClock } from '../state/format'
import { useConsole } from '../state/store'
import type {
  AuthorBuilding,
  DraftFire,
  DraftValidation,
  Preview,
  RecordAreasPayload,
  StoredPackage,
} from '../state/types'
import { Badge, Button, EmptyState, Panel, StatTile } from '../ui/primitives'
import { AuthorMap, lonLatToSim, type AuthorTool } from './AuthorMap'
import {
  addToSelection,
  areaColorByBuilding,
  areasFromPackage,
  areasFromRecord,
  AREA_PALETTE,
  assignAreaMembers,
  buildDraft,
  buildingsInBox,
  clearAreaMembers,
  fireRadiusAt,
  MAX_AGENTS_PER_BUILDING,
  newArea,
  newFire,
  orderedAreas,
  packageIdProblem,
  pruneAreas,
  RECORD_WAVES,
  removeArea,
  removeFromSelection,
  setAllCounts,
  setCount,
  suggestPackageId,
  toggleAreaMember,
  toggleBuilding,
  totals,
  unassignedAreas,
  unspawnableInBox,
  updateArea,
  type Box,
  type DraftAlertArea,
  type Household,
} from './selection'

const TOOLS: { id: AuthorTool; label: string; hint: string }[] = [
  { id: 'households', label: 'Households', hint: 'Drag a box over the buildings that should evacuate.' },
  { id: 'count', label: 'Agents', hint: 'Click a household to set how many agents it holds.' },
  { id: 'area', label: 'Alert areas', hint: 'Pick an area, then drag a box over the households its order covers.' },
  { id: 'fire', label: 'Fire origins', hint: 'Click where a fire starts, then set how it grows.' },
]

/**
 * Setting one building's agent count, anchored where the building is.
 *
 * Most buildings are houses holding one vehicle. A school or a care home holds many, and
 * finding that one building among several hundred is only practical on the map itself,
 * so the editor comes to the building rather than the other way round.
 */
function CountStepper({
  buildingId,
  count,
  position,
  onChange,
  onClose,
}: {
  buildingId: string
  count: number
  position: { x: number; y: number }
  onChange: (next: number) => void
  onClose: () => void
}) {
  const step = (delta: number) => onChange(count + delta)
  return (
    <div
      className="panel absolute z-20 w-44 -translate-x-1/2 -translate-y-full p-2 shadow-lg"
      style={{ left: position.x, top: position.y - 12 }}
      onPointerDown={(event) => event.stopPropagation()}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-micro text-ink-faint" title={buildingId}>
          building {buildingId}
        </span>
        <button type="button" className="text-micro text-ink-faint hover:text-ink-text" onClick={onClose}>
          close
        </button>
      </div>
      <div className="mt-1.5 flex items-center gap-1">
        <button type="button" className="btn btn-ghost px-2" onClick={() => step(-1)} disabled={count <= 1}>
          -
        </button>
        <input
          className="input tnum w-full text-center"
          type="number"
          min={1}
          max={MAX_AGENTS_PER_BUILDING}
          value={count}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        <button
          type="button"
          className="btn btn-ghost px-2"
          onClick={() => step(1)}
          disabled={count >= MAX_AGENTS_PER_BUILDING}
        >
          +
        </button>
      </div>
      <div className="mt-1.5 flex gap-1">
        {[1, 2, 5, 20].map((preset) => (
          <button
            key={preset}
            type="button"
            className={`flex-1 rounded border px-1 py-0.5 text-micro ${
              count === preset ? 'border-status-nominal text-ink-text' : 'border-ink-line text-ink-muted'
            }`}
            onClick={() => onChange(preset)}
          >
            {preset}
          </button>
        ))}
      </div>
    </div>
  )
}

function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: React.ReactNode
}) {
  return (
    <label className="block">
      <span className="text-small font-medium text-ink-muted">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-micro text-ink-faint">{hint}</span>}
    </label>
  )
}

function IssueList({ issues, tone }: { issues: { field: string; message: string; hint: string }[]; tone: 'bad' | 'warn' }) {
  if (issues.length === 0) return null
  return (
    <ul className="space-y-1.5">
      {issues.map((issue, i) => (
        <li key={`${issue.field}-${i}`} className="text-micro">
          <span className={tone === 'bad' ? 'text-status-hazard' : 'text-status-caution'}>
            {issue.message}
          </span>
          <span className="block text-ink-faint">{issue.hint}</span>
        </li>
      ))}
    </ul>
  )
}

function FireEditor({
  fire,
  onChange,
  onRemove,
}: {
  fire: DraftFire
  onChange: (next: DraftFire) => void
  onRemove: () => void
}) {
  const num = (key: keyof DraftFire) => (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = Number(event.target.value)
    onChange({ ...fire, [key]: Number.isFinite(value) ? value : 0 })
  }
  return (
    <div className="rounded border border-ink-line bg-ink-bg p-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-small font-medium">{fire.id}</span>
        <Button variant="ghost" onClick={onRemove}>
          Remove
        </Button>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2">
        <Field label="Starts at (s)">
          <input className="input tnum" type="number" value={fire.t0} onChange={num('t0')} />
        </Field>
        <Field label="Initial radius (m)">
          <input className="input tnum" type="number" value={fire.r0} onChange={num('r0')} />
        </Field>
        <Field label="Growth (m/s)">
          <input className="input tnum" type="number" step="0.1" value={fire.growth_m_per_s} onChange={num('growth_m_per_s')} />
        </Field>
        <Field label="Radius cap (m)">
          <input
            className="input tnum"
            type="number"
            value={fire.max_r_m ?? ''}
            placeholder="none"
            onChange={(event) =>
              onChange({ ...fire, max_r_m: event.target.value === '' ? null : Number(event.target.value) })
            }
          />
        </Field>
      </div>
    </div>
  )
}

/**
 * One alert area, with the order that covers it.
 *
 * The area being drawn into is the selected one, which is why the whole row is a button.
 * Its colour is what the map paints its households, so the swatch is the legend.
 */
function AreaEditor({
  area,
  selected,
  households,
  onSelect,
  onChange,
  onRemove,
  onOrderAll,
  onClearMembers,
}: {
  area: DraftAlertArea
  selected: boolean
  households: number
  onSelect: () => void
  onChange: (patch: Partial<DraftAlertArea>) => void
  onRemove: () => void
  onOrderAll: () => void
  onClearMembers: () => void
}) {
  const wave = RECORD_WAVES.find((w) => w.id === area.wave)
  const unassigned = area.issueTimeS == null
  const offRecord = wave != null && !unassigned && wave.issueTimeS !== area.issueTimeS
  return (
    <div className={`card-choice ${selected ? 'card-choice-on' : 'card-choice-off'} p-2`}>
      {/* The whole row selects, because selecting is what says where a drag lands. The
          name is text here and editable below, so the row has no interactive child
          competing for the click. */}
      <button
        type="button"
        className="flex w-full items-center gap-2 text-left"
        aria-pressed={selected}
        onClick={onSelect}
      >
        <span
          className="h-3.5 w-3.5 shrink-0 rounded-sm border border-ink-line"
          style={{ backgroundColor: area.color, opacity: unassigned ? 0.45 : 1 }}
        />
        <span className="min-w-0 flex-1 truncate text-small font-medium">{area.name}</span>
        {selected && <Badge tone="nominal">drawing here</Badge>}
        {unassigned && <Badge tone="caution">no order</Badge>}
        <span className="tnum shrink-0 text-micro text-ink-faint">{integer(area.members.length)}</span>
      </button>

      {selected && (
        <Field label="Area name">
          <input
            className="input mt-2"
            value={area.name}
            onChange={(event) => onChange({ name: event.target.value })}
          />
        </Field>
      )}

      <div className="mt-2 grid grid-cols-2 gap-2">
        <Field label="Wave">
          <select
            className="input"
            value={area.wave ?? (unassigned ? '' : 'custom')}
            onChange={(event) => {
              const value = event.target.value
              if (value === '') {
                onChange({ wave: null, issueTimeS: null })
                return
              }
              if (value === 'custom') {
                onChange({ wave: null, issueTimeS: area.issueTimeS ?? 6300 })
                return
              }
              const next = RECORD_WAVES.find((w) => w.id === value)
              if (next) onChange({ wave: next.id, issueTimeS: next.issueTimeS, color: next.color })
            }}
          >
            <option value="">no order</option>
            {RECORD_WAVES.map((w) => (
              <option key={w.id} value={w.id}>
                {w.id} &middot; {w.wallClock}
              </option>
            ))}
            <option value="custom">custom time</option>
          </select>
        </Field>
        <Field label="Issued at (s)" hint={unassigned ? 'no order' : simClock(area.issueTimeS as number)}>
          <input
            className="input tnum"
            type="number"
            value={area.issueTimeS ?? ''}
            placeholder="none"
            onChange={(event) =>
              onChange({
                issueTimeS: event.target.value === '' ? null : Number(event.target.value) || 0,
              })
            }
          />
        </Field>
      </div>

      {unassigned && (
        <p className="mt-1 text-micro text-ink-faint">
          No broadcast on 28 May named this area. Its {integer(area.members.length)} households run
          with no order until a wave is chosen.
        </p>
      )}

      {offRecord && (
        <p className="mt-1 text-micro text-status-caution">
          {area.wave} went out at {simClock(wave.issueTimeS)} in the record. This area is set to{' '}
          {simClock(area.issueTimeS as number)}.
        </p>
      )}

      <div className="mt-2 flex flex-wrap gap-1">
        {AREA_PALETTE.map((color) => (
          <button
            key={color}
            type="button"
            aria-label={`colour ${color}`}
            className={`h-4 w-4 rounded-sm border ${
              area.color === color ? 'border-ink-text' : 'border-ink-line'
            }`}
            style={{ backgroundColor: color }}
            onClick={() => onChange({ color })}
          />
        ))}
      </div>

      {selected && (
        <>
          <Field label="Message">
            <textarea
              className="input mt-2 h-16 resize-none"
              value={area.hazardText}
              onChange={(event) => onChange({ hazardText: event.target.value })}
            />
          </Field>
          <Field label="Comfort centre" hint="Named in the order, left empty when none was.">
            <input
              className="input"
              value={area.comfortCentre ?? ''}
              placeholder="none"
              onChange={(event) => onChange({ comfortCentre: event.target.value || null })}
            />
          </Field>
          <div className="mt-2 flex flex-wrap gap-2">
            <Button variant="ghost" onClick={onOrderAll} disabled={households === 0}>
              Order every unassigned household
            </Button>
            {area.members.length > 0 && (
              <Button variant="ghost" onClick={onClearMembers}>
                Empty
              </Button>
            )}
            <Button variant="ghost" onClick={onRemove}>
              Remove
            </Button>
          </div>
        </>
      )}
    </div>
  )
}

/**
 * Authoring a scenario package by drawing on the map.
 *
 * Four tools over one map. Households are the buildings that evacuate, alert areas are the
 * groups of households an order covers, each with its own issue time and colour, and fire
 * origins are placed by click. The package is validated on the backend as the draft
 * changes, so the operator sees a named problem while there is still something to change,
 * and creating it never writes over anything.
 */
export function AuthorView({ onDone }: { onDone: (packageId: string) => void }) {
  const pushToast = useConsole((s) => s.pushToast)

  const [sourcePackage, setSourcePackage] = useState<string>('')
  const [sources, setSources] = useState<string[]>([])
  const [buildings, setBuildings] = useState<AuthorBuilding[]>([])
  const [roads, setRoads] = useState<GeoJSON.FeatureCollection | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)

  const [tool, setTool] = useState<AuthorTool>('households')
  const [subtractive, setSubtractive] = useState(false)
  const [households, setHouseholds] = useState<Household[]>([])
  const [areas, setAreas] = useState<DraftAlertArea[]>([])
  const [selectedArea, setSelectedArea] = useState<string | null>(null)
  const [recordAreas, setRecordAreas] = useState<RecordAreasPayload | null>(null)
  const [stored, setStored] = useState<StoredPackage | null>(null)
  const [fires, setFires] = useState<DraftFire[]>([])
  const [selectedFire, setSelectedFire] = useState<string | null>(null)
  const [anchorId, setAnchorId] = useState<string | null>(null)
  const [anchorPos, setAnchorPos] = useState<{ x: number; y: number } | null>(null)
  const [previewTimeS, setPreviewTimeS] = useState(1800)

  const [label, setLabel] = useState('')
  const [packageId, setPackageId] = useState('')
  const [idTouched, setIdTouched] = useState(false)
  const [description, setDescription] = useState('')

  const [staleBundle, setStaleBundle] = useState(false)
  const [validation, setValidation] = useState<DraftValidation | null>(null)
  const [creating, setCreating] = useState(false)

  const index = useMemo(() => new Map(buildings.map((b) => [b.id, b])), [buildings])
  const householdIds = useMemo(() => new Set(households.map((h) => h.building_id)), [households])
  const areaColors = useMemo(() => areaColorByBuilding(areas), [areas])
  const counts = useMemo(
    () => new Map(households.map((h) => [h.building_id, h.count])),
    [households],
  )
  const raised = useMemo(() => households.filter((h) => h.count > 1).length, [households])
  const storedAgents = useMemo(
    () => (stored ? stored.households.reduce((sum, h) => sum + h.count, 0) : 0),
    [stored],
  )
  const ordered = useMemo(
    () => orderedAreas(areas).reduce((sum, a) => sum + a.members.length, 0),
    [areas],
  )
  // Every household an order does not reach, whether it sits in an area carrying no time
  // or in no area at all. Counting only the first would let a household dragged out of
  // every area vanish from both tallies while still running unordered.
  const unassignedCount = Math.max(0, households.length - ordered)
  const looseCount = useMemo(
    () =>
      unassignedCount - unassignedAreas(areas).reduce((sum, a) => sum + a.members.length, 0),
    [unassignedCount, areas],
  )

  // Every area is a subset of the households by construction, so removing a household
  // removes it from its area in the same breath.
  useEffect(() => {
    setAreas((current) => pruneAreas(current, householdIds))
  }, [householdIds])

  // Keep a selection pointing at an area that still exists, so the area tool always has
  // somewhere to draw into.
  useEffect(() => {
    if (areas.length === 0) {
      if (selectedArea !== null) setSelectedArea(null)
      return
    }
    if (!areas.some((a) => a.key === selectedArea)) setSelectedArea(areas[0].key)
  }, [areas, selectedArea])
  const counted = useMemo(() => totals(households, index), [households, index])

  // Holding shift turns a drag into a removal, which is the fastest way to trim a
  // selection that overshot.
  useEffect(() => {
    const down = (event: KeyboardEvent) => event.key === 'Shift' && setSubtractive(true)
    const up = (event: KeyboardEvent) => event.key === 'Shift' && setSubtractive(false)
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [])

  // The list of packages a draft can inherit its network and shelters from.
  useEffect(() => {
    api
      .packages()
      .then((payload) => {
        const usable = payload.packages.filter((p) => p.usable && p.preview_ready).map((p) => p.id)
        setSources(usable)
        if (usable.length > 0) setSourcePackage((current) => current || usable[0])
      })
      .catch(() => setLoadError('The console backend is not answering.'))
  }, [])

  // The building layer and the basemap for whichever source package is chosen.
  useEffect(() => {
    if (!sourcePackage) return
    let cancelled = false
    setLoading(true)
    setLoadError(null)
    Promise.all([api.packageBuildings(sourcePackage), api.packagePreview(sourcePackage)])
      .then(([layer, preview]: [{ buildings: AuthorBuilding[] }, Preview]) => {
        if (cancelled) return
        setBuildings(layer.buildings)
        setRoads(preview.roads ?? null)
        // Simulation coordinates arrived in the bundle after the first ones were built,
        // and a fire cannot be placed without them.
        setStaleBundle(layer.buildings.length > 0 && layer.buildings[0].x == null)
        setHouseholds([])
        setAreas([])
        setFires([])
      })
      .catch((error: unknown) => {
        if (cancelled) return
        setBuildings([])
        setRoads(null)
        setLoadError(
          error instanceof ApiError && error.status === 404
            ? `${sourcePackage} has no building layer. Run python -m ui.tools.build_map_assets ${sourcePackage}.`
            : 'The building layer could not be loaded.',
        )
      })
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [sourcePackage])

  const effectiveId = idTouched ? packageId : suggestPackageId(label)
  const idProblem = packageIdProblem(effectiveId)

  const draft = useMemo(
    () =>
      buildDraft({
        id: effectiveId,
        label,
        description,
        sourcePackage,
        households,
        areas,
        areaChannel: 'broadcast',
        areaInstruction: 'evacuate_now',
        fires,
      }),
    [effectiveId, label, description, sourcePackage, households, areas, fires],
  )

  // Validation is asked for as the draft settles, so a problem shows up while the
  // operator is still looking at the thing that caused it.
  const validateTimer = useRef<number | null>(null)
  useEffect(() => {
    if (!sourcePackage || households.length === 0) {
      setValidation(null)
      return
    }
    if (validateTimer.current) window.clearTimeout(validateTimer.current)
    validateTimer.current = window.setTimeout(() => {
      api
        .validatePackage(draft)
        .then(setValidation)
        .catch(() => setValidation(null))
    }, 400)
    return () => {
      if (validateTimer.current) window.clearTimeout(validateTimer.current)
    }
  }, [draft, sourcePackage, households.length])

  // ------------------------------------------------------------------ drawing
  const onBox = useCallback(
    (box: Box, subtract: boolean) => {
      const picked = buildingsInBox(buildings, box)
      if (tool === 'households') {
        const dropped = unspawnableInBox(buildings, box).length
        setHouseholds((current) =>
          subtract ? removeFromSelection(current, picked) : addToSelection(current, picked),
        )
        if (!subtract && dropped > 0) {
          pushToast(`${dropped} buildings in that box sit too far from a road to spawn`, 'warn')
        }
      } else if (tool === 'area') {
        if (!selectedArea) {
          pushToast('Add an alert area first, then draw the households it covers.', 'warn')
          return
        }
        setAreas((current) =>
          subtract
            ? clearAreaMembers(current, selectedArea, picked)
            : assignAreaMembers(current, selectedArea, picked, householdIds),
        )
        const skipped = picked.filter((b) => !householdIds.has(b.id)).length
        if (!subtract && skipped > 0) {
          pushToast(
            `${skipped} buildings in that box are not households, so the order skips them`,
            'warn',
          )
        }
      }
    },
    [buildings, tool, householdIds, selectedArea, pushToast],
  )

  const onBuildingClick = useCallback(
    (buildingId: string) => {
      const building = index.get(buildingId)
      if (!building) return
      if (tool === 'households') setHouseholds((current) => toggleBuilding(current, building))
      else if (tool === 'area') {
        if (!selectedArea) {
          pushToast('Add an alert area first, then click the households it covers.', 'warn')
          return
        }
        if (!householdIds.has(buildingId)) {
          pushToast('Only a household can be ordered. Select it under Households first.', 'warn')
          return
        }
        setAreas((current) => toggleAreaMember(current, selectedArea, buildingId, householdIds))
      }
      else if (tool === 'count') {
        // Only a household holds agents, so clicking anything else says so instead of
        // opening an editor that could not change anything.
        if (!households.some((h) => h.building_id === buildingId)) {
          pushToast('That building is not a household yet. Select it first.', 'warn')
          return
        }
        setAnchorId(buildingId)
      }
    },
    [index, tool, households, householdIds, pushToast],
  )

  const onPlaceFire = useCallback(
    (lon: number, lat: number) => {
      const sim = lonLatToSim(lon, lat, buildings)
      if (!sim) {
        pushToast(
          `Rebuild this package's map bundle first: python -m ui.tools.build_map_assets ${sourcePackage} --force`,
          'warn',
        )
        return
      }
      setFires((current) => {
        const fire = newFire(current.length, sim.x, sim.y)
        setSelectedFire(fire.id)
        return [...current, fire]
      })
    },
    [buildings, pushToast, sourcePackage],
  )

  // Whether this source package has had its households placed against the alert record.
  useEffect(() => {
    if (!sourcePackage) {
      setRecordAreas(null)
      return
    }
    let cancelled = false
    api
      .packageRecordAreas(sourcePackage)
      .then((payload) => !cancelled && setRecordAreas(payload))
      .catch(() => !cancelled && setRecordAreas(null))
    return () => {
      cancelled = true
    }
  }, [sourcePackage])

  const loadRecordAreas = useCallback(() => {
    if (!recordAreas) return
    const seeded = areasFromRecord(recordAreas, buildings)
    setHouseholds(seeded.households)
    setAreas(seeded.areas)
    setSelectedArea(seeded.areas[0]?.key ?? null)
    const unordered = seeded.households.length - seeded.areas.reduce((n, a) => n + a.members.length, 0)
    pushToast(
      `${integer(seeded.households.length)} households in ${seeded.areas.length} ordered areas` +
        (unordered > 0 ? `, ${integer(unordered)} left unordered by the record` : ''),
      'good',
    )
    if (seeded.skipped > 0) {
      pushToast(`${integer(seeded.skipped)} record households are not in this map bundle`, 'warn')
    }
  }, [recordAreas, buildings, pushToast])

  // What the source package already holds, so it can be reopened instead of redrawn.
  useEffect(() => {
    if (!sourcePackage) {
      setStored(null)
      return
    }
    let cancelled = false
    api
      .packageAuthoring(sourcePackage)
      .then((payload) => {
        if (cancelled) return
        // A package with no buildings to redraw has nothing to reopen, even though its
        // fires are still readable through the fire panel.
        setStored(payload.households.length > 0 ? payload : null)
      })
      .catch(() => !cancelled && setStored(null))
    return () => {
      cancelled = true
    }
  }, [sourcePackage])

  const loadStored = useCallback(() => {
    if (!stored) return
    const opened = areasFromPackage(stored, buildings)
    setHouseholds(opened.households)
    setAreas(opened.areas)
    setSelectedArea(opened.areas[0]?.key ?? null)
    if (stored.fires.length > 0) {
      setFires(stored.fires)
      setSelectedFire(stored.fires[0]?.id ?? null)
    }
    pushToast(
      `Loaded ${integer(opened.households.length)} households, ${opened.areas.length} areas ` +
        `and ${stored.fires.length} fire origins from ${sourcePackage}`,
      'good',
    )
    if (opened.skipped > 0) {
      pushToast(`${integer(opened.skipped)} of its households are not in this map bundle`, 'warn')
    }
  }, [stored, buildings, sourcePackage, pushToast])

  const importFires = useCallback(
    async (packageId: string) => {
      if (!packageId) return
      try {
        const payload = await api.packageFires(packageId)
        if (payload.fires.length === 0) {
          pushToast(`${packageId} has no fire origin to take`, 'warn')
          return
        }
        setFires(payload.fires)
        setSelectedFire(payload.fires[0]?.id ?? null)
        pushToast(`Took ${integer(payload.fires.length)} fire origins from ${packageId}`, 'good')
      } catch {
        pushToast(`${packageId} has no fires.json to read`, 'bad')
      }
    },
    [pushToast],
  )

  const create = async () => {
    setCreating(true)
    try {
      const result = await api.createPackage(draft)
      pushToast(`Created ${result.package}. Build its map bundle to see it in Setup.`, 'good')
      onDone(result.package ?? effectiveId)
    } catch (error) {
      if (error instanceof ApiError && error.payload) {
        setValidation(error.payload as DraftValidation)
        pushToast(
          error.status === 409 ? 'That package name is already taken' : 'The package was not created',
          'bad',
        )
      } else {
        pushToast('The package was not created', 'bad')
      }
    } finally {
      setCreating(false)
    }
  }

  const blocked = Boolean(idProblem) || households.length === 0 || fires.length === 0 || !validation?.ok
  const activeTool = TOOLS.find((t) => t.id === tool)

  return (
    <div className="grid h-full min-h-0 grid-cols-[380px_minmax(0,1fr)] gap-3 p-3">
      <div className="flex min-h-0 flex-col gap-3 overflow-auto">
        <Panel title="Package" bodyClassName="p-0">
          <div className="space-y-3 p-3">
            <Field
              label="Source package"
              hint="Its road network, shelters, and routes are inherited. Choosing one gives you its map and an empty canvas."
            >
              <select
                className="input"
                value={sourcePackage}
                onChange={(event) => setSourcePackage(event.target.value)}
              >
                {sources.length === 0 && <option value="">no package with a map bundle</option>}
                {sources.map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </select>
            </Field>

            {stored && (
              <div className="rounded border border-ink-line p-2">
                <p className="text-micro text-ink-faint">
                  {sourcePackage} already holds {integer(storedAgents)} agents in{' '}
                  {integer(stored.households.length)} households, {stored.areas.length}{' '}
                  {stored.areas.length === 1 ? 'alert area' : 'alert areas'}, and{' '}
                  {stored.fires.length} fire origins. Load them to revise the package and save
                  the result under a new name.
                </p>
                <Button variant="ghost" onClick={loadStored}>
                  Load its households, areas and fires
                </Button>
                {stored.areas_without_buildings.length > 0 && (
                  <p className="mt-1 text-micro text-status-caution">
                    {stored.areas_without_buildings.join(', ')}{' '}
                    {stored.areas_without_buildings.length === 1 ? 'is an edge list' : 'are edge lists'}{' '}
                    with no buildings, so {stored.areas_without_buildings.length === 1 ? 'it' : 'they'}{' '}
                    cannot be redrawn here.
                  </p>
                )}
              </div>
            )}
            <Field label="Name shown in Setup">
              <input
                className="input"
                value={label}
                placeholder="Westwood ignition area"
                onChange={(event) => setLabel(event.target.value)}
              />
            </Field>
            <Field label="Directory name" hint={idProblem ?? 'Created under configs/, never overwriting.'}>
              <input
                className="input"
                value={effectiveId}
                onChange={(event) => {
                  setIdTouched(true)
                  setPackageId(event.target.value)
                }}
              />
            </Field>
            <Field label="Note" hint="Written into the package README.">
              <textarea
                className="input h-16 resize-none"
                value={description}
                onChange={(event) => setDescription(event.target.value)}
              />
            </Field>
          </div>
        </Panel>

        <Panel title="Drawing" bodyClassName="p-0">
          <div className="space-y-3 p-3">
            {staleBundle && (
              <p className="text-micro text-status-caution">
                This package&rsquo;s map bundle predates simulation coordinates, so fire origins
                cannot be placed. Rebuild it with python -m ui.tools.build_map_assets{' '}
                {sourcePackage} --force
              </p>
            )}
            <div className="flex gap-1">
              {TOOLS.map((entry) => (
                <button
                  key={entry.id}
                  type="button"
                  disabled={entry.id === 'fire' && staleBundle}
                  onClick={() => {
                    setTool(entry.id)
                    setAnchorId(null)
                  }}
                  className={`card-choice flex-1 text-center ${
                    tool === entry.id ? 'card-choice-on' : 'card-choice-off'
                  }`}
                >
                  <span className="text-small font-medium">{entry.label}</span>
                </button>
              ))}
            </div>
            <p className="text-micro text-ink-faint">
              {activeTool?.hint} Hold shift and drag to remove. Pan with the right or middle
              button, zoom with the wheel.
            </p>

            <div className="grid grid-cols-3 gap-2">
              <StatTile label="Buildings" value={integer(counted.buildings)} />
              <StatTile label="Agents" value={integer(counted.agents)} />
              <StatTile label="Roads" value={integer(counted.roads)} />
            </div>

            {tool === 'count' && (
              <p className="text-micro text-ink-faint">
                {raised === 0
                  ? 'Every household holds one agent. Click one on the map to raise it.'
                  : `${integer(raised)} buildings hold more than one agent, ringed in white on the map.`}
              </p>
            )}

            {tool === 'households' && households.length > 0 && (
              <Field
                label="Set every household at once"
                hint="Overwrites any per-building count already set. Use the Agents tool for one building."
              >
                <input
                  className="input tnum"
                  type="number"
                  min={1}
                  max={MAX_AGENTS_PER_BUILDING}
                  defaultValue={1}
                  onChange={(event) => setHouseholds((c) => setAllCounts(c, Number(event.target.value)))}
                />
              </Field>
            )}

            {households.length > 0 && (
              <Button variant="ghost" onClick={() => setHouseholds([])}>
                Clear households
              </Button>
            )}
          </div>
        </Panel>

        {tool === 'area' && (
          <Panel
            title="Alert areas"
            bodyClassName="p-0"
            action={
              <Button variant="ghost" onClick={() => {
                const area = newArea(areas)
                setAreas((current) => [...current, area])
                setSelectedArea(area.key)
              }}>
                Add area
              </Button>
            }
          >
            <div className="space-y-3 p-3">
              <div className="grid grid-cols-3 gap-2">
                <StatTile label="Ordered" value={integer(ordered)} />
                <StatTile label="No order" value={integer(unassignedCount)} />
                <StatTile label="Areas" value={integer(areas.length)} />
              </div>
              <p className="text-micro text-ink-faint">
                Each area draws in its own colour, and an order is scoped to the roads its
                households sit on. Areas sharing an issue time go out as one broadcast. An area
                marked no order holds its households and writes nothing, so give it a wave to
                bring it into the schedule, or drag its households into another area.
              </p>
              {looseCount > 0 && (
                <p className="text-micro text-status-caution">
                  {integer(looseCount)} households belong to no area at all, shown blue. Draw them
                  into an area, or leave them to run with no order.
                </p>
              )}

              {households.length === 0 && (
                <p className="text-micro text-status-caution">
                  Select households first. An order only reaches buildings that hold agents.
                </p>
              )}

              {recordAreas && (
                <div className="rounded border border-ink-line p-2">
                  <p className="text-micro text-ink-faint">
                    This map has {recordAreas.areas.length} communities placed against the 2023
                    alert record, covering {integer(recordAreas.households)} households.
                  </p>
                  <Button variant="ghost" onClick={loadRecordAreas}>
                    Load record areas
                  </Button>
                  <p className="mt-1 text-micro text-ink-faint">
                    Replaces the current households and areas with the record placement.
                  </p>
                </div>
              )}

              {areas.length === 0 && (
                <p className="text-micro text-ink-faint">
                  No area yet. Add one, then drag a box over the households its order covers.
                </p>
              )}

              <div className="space-y-2">
                {areas.map((area) => (
                  <AreaEditor
                    key={area.key}
                    area={area}
                    selected={area.key === selectedArea}
                    households={households.length}
                    onSelect={() => setSelectedArea(area.key)}
                    onChange={(patch) => setAreas((current) => updateArea(current, area.key, patch))}
                    onRemove={() => setAreas((current) => removeArea(current, area.key))}
                    onOrderAll={() =>
                      setAreas((current) =>
                        assignAreaMembers(
                          current,
                          area.key,
                          households
                            .filter((h) => !areaColors.has(h.building_id))
                            .map((h) => index.get(h.building_id))
                            .filter((b): b is NonNullable<typeof b> => Boolean(b)),
                          householdIds,
                        ),
                      )
                    }
                    onClearMembers={() =>
                      setAreas((current) => updateArea(current, area.key, { members: [] }))
                    }
                  />
                ))}
              </div>
            </div>
          </Panel>
        )}

        {tool === 'fire' && (
          <Panel title="Fire origins" bodyClassName="p-0">
            <div className="space-y-3 p-3">
              {fires.length === 0 && (
                <p className="text-micro text-ink-faint">Click the map to place the first origin.</p>
              )}

              <div className="rounded border border-ink-line p-2">
                <Field
                  label="Take the fires from another package"
                  hint="Replaces what is placed. Use it for a record-exact fire, whose coordinates cannot be clicked."
                >
                  <select
                    className="input"
                    value=""
                    onChange={(event) => importFires(event.target.value)}
                  >
                    <option value="">choose a package</option>
                    {sources.map((id) => (
                      <option key={id} value={id}>
                        {id}
                      </option>
                    ))}
                  </select>
                </Field>
              </div>

              <Field label={`Front shown at ${simClock(previewTimeS)}`}>
                <input
                  type="range"
                  min={0}
                  max={28800}
                  step={60}
                  value={previewTimeS}
                  className="w-full"
                  onChange={(event) => setPreviewTimeS(Number(event.target.value))}
                />
              </Field>
              <div className="space-y-2">
                {fires.map((fire) => (
                  <div key={fire.id} onFocus={() => setSelectedFire(fire.id)}>
                    <FireEditor
                      fire={fire}
                      onChange={(next) =>
                        setFires((current) => current.map((f) => (f.id === fire.id ? next : f)))
                      }
                      onRemove={() => setFires((current) => current.filter((f) => f.id !== fire.id))}
                    />
                    <p className="tnum mt-1 text-micro text-ink-faint">
                      radius {integer(fireRadiusAt(fire, previewTimeS))} m at {simClock(previewTimeS)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </Panel>
        )}

        <Panel title="Before it is created" bodyClassName="p-0">
          <div className="space-y-3 p-3">
            {validation && (
              <>
                <IssueList issues={validation.problems} tone="bad" />
                <IssueList issues={validation.warnings} tone="warn" />
                {validation.ok && validation.problems.length === 0 && (
                  <p className="text-micro text-status-nominal">Ready to create.</p>
                )}
              </>
            )}
            {households.length === 0 && (
              <p className="text-micro text-ink-faint">Select at least one household.</p>
            )}
            {fires.length === 0 && (
              <p className="text-micro text-ink-faint">Place at least one fire origin.</p>
            )}
            <Button variant="primary" onClick={create} busy={creating} disabled={blocked}>
              Create package
            </Button>
            <p className="text-micro text-ink-faint">
              Creating writes a new directory under configs/. An existing package is never changed.
            </p>
          </div>
        </Panel>
      </div>

      <Panel
        title="Map"
        bodyClassName="min-h-0 flex-1 p-0"
        action={
          <div className="flex items-center gap-2">
            {subtractive && <Badge tone="caution">removing</Badge>}
            <span className="text-micro text-ink-faint">{integer(buildings.length)} buildings</span>
          </div>
        }
      >
        <div className="relative h-full w-full">
          {loadError ? (
            <div className="flex h-full items-center justify-center p-6">
              <EmptyState title="No building layer" detail={loadError} />
            </div>
          ) : loading ? (
            <div className="flex h-full items-center justify-center text-ink-muted">
              Loading the building layer
            </div>
          ) : (
            <AuthorMap
              buildings={buildings}
              roads={roads}
              householdIds={householdIds}
              areaColors={areaColors}
              counts={counts}
              anchorId={tool === 'count' ? anchorId : null}
              onAnchorMove={setAnchorPos}
              fires={fires}
              selectedFire={selectedFire}
              previewTimeS={previewTimeS}
              tool={tool}
              subtractive={subtractive}
              onBox={onBox}
              onBuildingClick={onBuildingClick}
              onPlaceFire={onPlaceFire}
              onSelectFire={setSelectedFire}
            />
          )}
          {tool === 'count' && anchorId && anchorPos && (
            <CountStepper
              buildingId={anchorId}
              count={counts.get(anchorId) ?? 1}
              position={anchorPos}
              onChange={(next) => setHouseholds((current) => setCount(current, anchorId, next))}
              onClose={() => setAnchorId(null)}
            />
          )}
        </div>
      </Panel>
    </div>
  )
}

export { setCount }
