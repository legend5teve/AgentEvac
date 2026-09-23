import type {
  AuthorBuilding,
  DraftFire,
  PackageDraft,
  RecordAreasPayload,
  StoredPackage,
} from '../state/types'

// Selecting households, selecting the area an order covers, and placing fire origins are
// all the same shape of decision, so the rules live here as plain functions the view
// calls. Nothing in this file touches MapLibre or React, which is what lets the parts an
// operator can get wrong be tested directly.

/** A drawn box, in longitude and latitude, in the order the two corners were clicked. */
export interface Box {
  lon1: number
  lat1: number
  lon2: number
  lat2: number
}

/** A household in the draft, meaning one selected building and how many agents it holds. */
export interface Household {
  building_id: string
  count: number
}

export const DEFAULT_AGENTS_PER_BUILDING = 1
export const MAX_AGENTS_PER_BUILDING = 50

/** Normalise a drawn box, since an operator may drag from any corner. */
export function normaliseBox(box: Box): { minLon: number; minLat: number; maxLon: number; maxLat: number } {
  return {
    minLon: Math.min(box.lon1, box.lon2),
    minLat: Math.min(box.lat1, box.lat2),
    maxLon: Math.max(box.lon1, box.lon2),
    maxLat: Math.max(box.lat1, box.lat2),
  }
}

/**
 * Buildings a drawn box selects.
 *
 * Selection is by centroid, so a building is in or out as a whole and an operator never
 * has to reason about a footprint straddling the edge of the box. Buildings with no road
 * to spawn onto are excluded here, because a household cannot be placed on one, and
 * leaving them in would mean the count on screen disagreed with the package written.
 */
export function buildingsInBox(buildings: AuthorBuilding[], box: Box): AuthorBuilding[] {
  const { minLon, minLat, maxLon, maxLat } = normaliseBox(box)
  return buildings.filter(
    (b) => Boolean(b.edge) && b.lon >= minLon && b.lon <= maxLon && b.lat >= minLat && b.lat <= maxLat,
  )
}

/** Buildings inside the box that cannot spawn, so the view can say how many were dropped. */
export function unspawnableInBox(buildings: AuthorBuilding[], box: Box): AuthorBuilding[] {
  const { minLon, minLat, maxLon, maxLat } = normaliseBox(box)
  return buildings.filter(
    (b) => !b.edge && b.lon >= minLon && b.lon <= maxLon && b.lat >= minLat && b.lat <= maxLat,
  )
}

/**
 * Add a box's buildings to the current households, keeping any count already edited.
 *
 * Boxes accumulate, so an operator can build a selection out of several draws without
 * losing the counts they set on an earlier one.
 */
export function addToSelection(current: Household[], picked: AuthorBuilding[]): Household[] {
  const byId = new Map(current.map((h) => [h.building_id, h]))
  for (const building of picked) {
    if (!byId.has(building.id)) {
      byId.set(building.id, { building_id: building.id, count: DEFAULT_AGENTS_PER_BUILDING })
    }
  }
  return [...byId.values()]
}

/** Remove a box's buildings from the current households. */
export function removeFromSelection(current: Household[], picked: AuthorBuilding[]): Household[] {
  const drop = new Set(picked.map((b) => b.id))
  return current.filter((h) => !drop.has(h.building_id))
}

/** Add one building if absent, remove it if present, which is what a click does. */
export function toggleBuilding(current: Household[], building: AuthorBuilding): Household[] {
  if (!building.edge) return current
  const present = current.some((h) => h.building_id === building.id)
  return present
    ? current.filter((h) => h.building_id !== building.id)
    : [...current, { building_id: building.id, count: DEFAULT_AGENTS_PER_BUILDING }]
}

/** Set one building's agent count, clamped to what the backend will accept. */
export function setCount(current: Household[], buildingId: string, count: number): Household[] {
  const clamped = Math.max(1, Math.min(MAX_AGENTS_PER_BUILDING, Math.round(count || 0)))
  return current.map((h) => (h.building_id === buildingId ? { ...h, count: clamped } : h))
}

/** Set every selected building's count at once, which is the usual way to scale a draft. */
export function setAllCounts(current: Household[], count: number): Household[] {
  const clamped = Math.max(1, Math.min(MAX_AGENTS_PER_BUILDING, Math.round(count || 0)))
  return current.map((h) => ({ ...h, count: clamped }))
}

// An alert area only ever covers buildings that are already households. A building with
// no agents cannot be evacuated, so ordering it changes nothing, and its road would be
// pulled into the ordered area anyway, which would over-scope the order onto households
// that were never selected. Holding the area to a subset of the households also means a
// coloured building on the map is necessarily a selected household, so the two states can
// never be confused for one another.
//
// A household belongs to at most one area. The 2023 waves were disjoint, each extension
// naming communities the earlier orders had not covered, and exclusivity is also what
// lets one colour on the map answer which order reaches a building.

/**
 * One alert area being authored, with the order that covers it.
 *
 * `issueTimeS` is null for an area that holds households and carries no order. The record
 * produces several of those, because communities inside the study area were named in no
 * broadcast, and an operator needs to see them to decide where they belong. Such an area
 * writes no broadcast, so its households run unordered until a time is chosen.
 */
export interface DraftAlertArea {
  /** Stable identity, so renaming an area never loses the buildings drawn into it. */
  key: string
  name: string
  color: string
  /** The record wave this area belongs to, or null when the time was typed by hand. */
  wave: string | null
  issueTimeS: number | null
  hazardText: string
  comfortCentre: string | null
  members: string[]
}

/**
 * The four broadcasts of 28 May 2023, as issue times a wave can be set from.
 *
 * These mirror `ui/assets/record_areas/_schedule.json`, which is the provenance-carrying
 * copy. They are repeated here so the wave menu works for a package that has no
 * record-area file, meaning any map outside the Halifax study area.
 */
export const RECORD_WAVES: { id: string; issueTimeS: number; wallClock: string; color: string }[] = [
  { id: 'EA-1', issueTimeS: 6300, wallClock: '17:13', color: '#E69F00' },
  { id: 'EA-2', issueTimeS: 9660, wallClock: '18:09', color: '#F0E442' },
  { id: 'EA-3', issueTimeS: 15180, wallClock: '19:41', color: '#CC79A7' },
  { id: 'EA-4', issueTimeS: 17460, wallClock: '20:19', color: '#8E5FA8' },
]

/**
 * Colours an added area cycles through.
 *
 * Chosen from the Okabe-Ito set the rest of the console uses, minus the vermilion the
 * fire front owns and the blue an unordered household is drawn in, so a wave can never be
 * mistaken for either.
 */
export const AREA_PALETTE = [
  '#E69F00', '#F0E442', '#CC79A7', '#8E5FA8',
  '#56B4E9', '#009E73', '#C2255C', '#B26A00',
]

let areaSeq = 0

/**
 * A new empty area, taking the next unused colour.
 *
 * It starts with no order, so the wave is a decision the operator makes rather than a
 * default that could write a broadcast nobody chose.
 */
export function newArea(existing: DraftAlertArea[]): DraftAlertArea {
  const used = new Set(existing.map((a) => a.color))
  const color = AREA_PALETTE.find((c) => !used.has(c)) ?? AREA_PALETTE[existing.length % AREA_PALETTE.length]
  areaSeq += 1
  return {
    key: `area_${areaSeq}`,
    name: `area_${existing.length + 1}`,
    color,
    wave: null,
    issueTimeS: null,
    hazardText: 'Evacuate immediately.',
    comfortCentre: null,
    members: [],
  }
}

/** Replace one area, leaving the rest untouched. */
export function updateArea(
  areas: DraftAlertArea[],
  key: string,
  patch: Partial<DraftAlertArea>,
): DraftAlertArea[] {
  return areas.map((area) => (area.key === key ? { ...area, ...patch } : area))
}

/** Drop one area entirely, releasing the households it held. */
export function removeArea(areas: DraftAlertArea[], key: string): DraftAlertArea[] {
  return areas.filter((area) => area.key !== key)
}

/**
 * Put households into one area and out of every other.
 *
 * Drawing a box over ground another area already claimed moves those households, which is
 * how an operator corrects a boundary without first having to clear the neighbour.
 */
export function assignAreaMembers(
  areas: DraftAlertArea[],
  key: string,
  picked: AuthorBuilding[],
  households: Set<string>,
): DraftAlertArea[] {
  const claim = new Set(picked.map((b) => b.id).filter((id) => households.has(id)))
  if (claim.size === 0) return areas
  return areas.map((area) => {
    if (area.key !== key) {
      const kept = area.members.filter((id) => !claim.has(id))
      return kept.length === area.members.length ? area : { ...area, members: kept }
    }
    const seen = new Set(area.members)
    const members = [...area.members]
    for (const building of picked) {
      if (claim.has(building.id) && !seen.has(building.id)) {
        seen.add(building.id)
        members.push(building.id)
      }
    }
    return { ...area, members }
  })
}

/** Take a box's buildings out of one area. */
export function clearAreaMembers(
  areas: DraftAlertArea[],
  key: string,
  picked: AuthorBuilding[],
): DraftAlertArea[] {
  const drop = new Set(picked.map((b) => b.id))
  return updateArea(areas, key, {
    members: (areas.find((a) => a.key === key)?.members ?? []).filter((id) => !drop.has(id)),
  })
}

/** Add one household to an area, or take it out if that area already holds it. */
export function toggleAreaMember(
  areas: DraftAlertArea[],
  key: string,
  buildingId: string,
  households: Set<string>,
): DraftAlertArea[] {
  if (!households.has(buildingId)) return areas
  const area = areas.find((a) => a.key === key)
  if (area?.members.includes(buildingId)) {
    return updateArea(areas, key, { members: area.members.filter((id) => id !== buildingId) })
  }
  return assignAreaMembers(areas, key, [{ id: buildingId } as AuthorBuilding], households)
}

/**
 * Drop area members that are no longer households.
 *
 * Deselecting a household has to take it out of its area too, otherwise the area would
 * quietly keep ordering a building that no longer holds anyone.
 */
export function pruneAreas(areas: DraftAlertArea[], households: Set<string>): DraftAlertArea[] {
  let changed = false
  const next = areas.map((area) => {
    const members = area.members.filter((id) => households.has(id))
    if (members.length === area.members.length) return area
    changed = true
    return { ...area, members }
  })
  return changed ? next : areas
}

/** Which colour each ordered building draws in, for the map. */
export function areaColorByBuilding(areas: DraftAlertArea[]): Map<string, string> {
  const out = new Map<string, string>()
  for (const area of areas) {
    for (const id of area.members) out.set(id, area.color)
  }
  return out
}

/** Areas that will write a broadcast, in the order they go out. */
export function orderedAreas(areas: DraftAlertArea[]): DraftAlertArea[] {
  return areas
    .filter((area) => area.members.length > 0 && area.name.trim().length > 0 && area.issueTimeS != null)
    .sort((a, b) => (a.issueTimeS as number) - (b.issueTimeS as number))
}

/**
 * Areas holding households that no order covers.
 *
 * These are what an operator still has to decide about, so the panel counts them and the
 * summary says how many households would run with no order.
 */
export function unassignedAreas(areas: DraftAlertArea[]): DraftAlertArea[] {
  return areas.filter((area) => area.members.length > 0 && area.issueTimeS == null)
}

export interface SelectionTotals {
  buildings: number
  agents: number
  roads: number
}

/** What the header counts while an operator is drawing. */
export function totals(households: Household[], index: Map<string, AuthorBuilding>): SelectionTotals {
  const roads = new Set<string>()
  let agents = 0
  for (const household of households) {
    agents += household.count
    const edge = index.get(household.building_id)?.edge
    if (edge) roads.add(edge)
  }
  return { buildings: households.length, agents, roads: roads.size }
}

/**
 * A package name the backend will take.
 *
 * The same rule the backend enforces, applied as the operator types, so a long draft is
 * never rejected at the end for something that could have been said at the start.
 */
export function packageIdProblem(id: string): string | null {
  if (!id) return 'Give the package a name.'
  if (!/^[a-z0-9]/.test(id)) return 'Start the name with a lowercase letter or a digit.'
  if (id.length < 3) return 'Use at least three characters.'
  if (id.length > 64) return 'Keep the name to 64 characters or fewer.'
  if (!/^[a-z0-9_]+$/.test(id)) return 'Use lowercase letters, digits, and underscores only.'
  return null
}

/** Turn a typed label into a usable package name, which is what the field suggests. */
export function suggestPackageId(label: string): string {
  return label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 64)
}

export interface DraftInput {
  id: string
  label: string
  description: string
  sourcePackage: string
  households: Household[]
  areas: DraftAlertArea[]
  areaChannel: string
  areaInstruction: string
  fires: DraftFire[]
}

/**
 * Assemble what the operator drew into the body the backend takes.
 *
 * Areas sharing an issue time become one broadcast, because that is what the record is.
 * EA-3 named Haliburton Hills and Glen Arbour in a single alert, and writing them as two
 * events at the same instant would claim two broadcasts where there was one. Events are
 * numbered in time order, so the ids a package carries are the order they went out.
 *
 * An area with no members is dropped, because an area covering nothing is rejected on the
 * far side, and the absence of a schedule is a legitimate package.
 */
export function buildDraft(input: DraftInput): PackageDraft {
  const draft: PackageDraft = {
    id: input.id.trim(),
    label: input.label.trim(),
    description: input.description.trim(),
    source_package: input.sourcePackage,
    households: input.households.map((h) => ({ building_id: h.building_id, count: h.count })),
    fires: input.fires,
  }

  const live = orderedAreas(input.areas)
  if (live.length === 0) return draft

  draft.alert_areas = live.map((area) => ({
    name: area.name.trim(),
    label: area.name.trim(),
    building_ids: area.members,
    color: area.color,
  }))

  // `live` already dropped every area with no issue time, so each one has a number here.
  const byTime = new Map<number, DraftAlertArea[]>()
  for (const area of live) {
    const issueTimeS = area.issueTimeS as number
    const bucket = byTime.get(issueTimeS)
    if (bucket) bucket.push(area)
    else byTime.set(issueTimeS, [area])
  }
  draft.alert_events = [...byTime.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([issueTimeS, group], index) => ({
      id: group[0].wave ?? `EA-${index + 1}`,
      issue_time_s: issueTimeS,
      areas: group.map((area) => area.name.trim()),
      instruction: input.areaInstruction,
      channel: input.areaChannel,
      // One broadcast carries one message, so the first area in the group supplies it.
      hazard_text: group[0].hazardText.trim(),
      comfort_centre: group[0].comfortCentre || null,
    }))
  return draft
}

/**
 * Turn the record placement into a starting selection.
 *
 * Every community becomes an area, whether or not a broadcast named it. The ones a
 * broadcast named carry that broadcast's time, colour and wording. The ones it did not
 * carry no time, so they hold their households, draw in their own colour, and write no
 * order until somebody gives them one. Showing them is the point, because deciding where
 * those households belong is a judgement the record cannot settle on its own.
 *
 * Building ids the loaded layer does not hold are dropped, which is what happens when the
 * record file was built for a different source package.
 */
export function areasFromRecord(
  payload: RecordAreasPayload,
  buildings: AuthorBuilding[],
): { households: Household[]; areas: DraftAlertArea[]; skipped: number } {
  const spawnable = new Set(buildings.filter((b) => Boolean(b.edge)).map((b) => b.id))
  const households: Household[] = []
  const areas: DraftAlertArea[] = []
  const seen = new Set<string>()
  let skipped = 0

  for (const area of payload.areas) {
    const members: string[] = []
    for (const id of area.building_ids) {
      if (!spawnable.has(id)) {
        skipped += 1
        continue
      }
      if (seen.has(id)) continue
      seen.add(id)
      households.push({ building_id: id, count: DEFAULT_AGENTS_PER_BUILDING })
      members.push(id)
    }
    if (members.length === 0) continue
    areaSeq += 1
    areas.push({
      key: `area_${areaSeq}`,
      name: area.name,
      color: area.color,
      wave: area.wave,
      issueTimeS: area.ordered ? area.issue_time_s : null,
      hazardText: area.hazard_text || 'Evacuate immediately.',
      comfortCentre: area.comfort_centre ?? null,
      members,
    })
  }
  // Ordered areas first, in broadcast order, then the ones still to be decided.
  areas.sort((a, b) => {
    if (a.issueTimeS == null && b.issueTimeS == null) return b.members.length - a.members.length
    if (a.issueTimeS == null) return 1
    if (b.issueTimeS == null) return -1
    return a.issueTimeS - b.issueTimeS
  })
  return { households, areas, skipped }
}

/**
 * Turn an existing package back into a draft selection.
 *
 * Choosing a source package supplies the map and clears the canvas, which is right for
 * drawing something new. Reopening is the other case, where a package is adjusted and
 * saved under a new name, and this is what that reads back.
 *
 * Buildings the loaded layer does not hold are dropped, so a package read against the
 * wrong map degrades to whatever it has in common rather than failing outright.
 */
export function areasFromPackage(
  payload: StoredPackage,
  buildings: AuthorBuilding[],
): { households: Household[]; areas: DraftAlertArea[]; skipped: number } {
  const spawnable = new Set(buildings.filter((b) => Boolean(b.edge)).map((b) => b.id))
  const households: Household[] = []
  let skipped = 0
  for (const row of payload.households) {
    if (!spawnable.has(row.building_id)) {
      skipped += 1
      continue
    }
    households.push({
      building_id: row.building_id,
      count: Math.max(1, Math.min(MAX_AGENTS_PER_BUILDING, Math.round(row.count || 1))),
    })
  }

  const held = new Set(households.map((h) => h.building_id))
  const areas: DraftAlertArea[] = []
  for (const area of payload.areas) {
    const members = area.building_ids.filter((id) => held.has(id))
    if (members.length === 0) continue
    areaSeq += 1
    areas.push({
      key: `area_${areaSeq}`,
      name: area.name,
      color: area.color,
      wave: area.wave,
      issueTimeS: area.issue_time_s,
      hazardText: area.hazard_text || 'Evacuate immediately.',
      comfortCentre: area.comfort_centre ?? null,
      members,
    })
  }
  areas.sort((a, b) => {
    if (a.issueTimeS == null && b.issueTimeS == null) return 0
    if (a.issueTimeS == null) return 1
    if (b.issueTimeS == null) return -1
    return a.issueTimeS - b.issueTimeS
  })
  return { households, areas, skipped }
}

/** A fire origin with the defaults the record's own sources use. */
export function newFire(index: number, x: number, y: number): DraftFire {
  return {
    id: `source_${index + 1}`,
    x: Math.round(x * 100) / 100,
    y: Math.round(y * 100) / 100,
    t0: 0,
    r0: 30,
    growth_m_per_s: 0.3,
    max_r_m: 700,
  }
}

/**
 * Radius of a fire at a given instant, the same growth the simulator applies.
 *
 * The scrubber uses this to show what the front covers at a chosen moment, so a placed
 * source can be judged against the households it would reach before the package is run.
 */
export function fireRadiusAt(fire: DraftFire, simTimeS: number): number {
  if (simTimeS < fire.t0) return 0
  const grown = fire.r0 + fire.growth_m_per_s * (simTimeS - fire.t0)
  if (fire.max_r_m != null) return Math.max(0, Math.min(grown, fire.max_r_m))
  return Math.max(0, grown)
}
