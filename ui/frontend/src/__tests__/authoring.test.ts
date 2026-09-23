import { validateStyleMin } from '@maplibre/maplibre-gl-style-spec'
import { describe, expect, it } from 'vitest'
import { boxFeature, buildAuthorStyle, buildingFeature, squareRing } from '../authoring/authorStyle'
import {
  addToSelection,
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
  normaliseBox,
  orderedAreas,
  packageIdProblem,
  pruneAreas,
  unassignedAreas,
  removeFromSelection,
  setAllCounts,
  setCount,
  suggestPackageId,
  toggleAreaMember,
  toggleBuilding,
  totals,
  unspawnableInBox,
  type DraftAlertArea,
} from '../authoring/selection'
import type { AuthorBuilding } from '../state/types'

function b(id: string, lon: number, lat: number, edge: string | null = 'e0'): AuthorBuilding {
  return { id, lon, lat, edge, edge_dist_m: edge ? 12 : null, x: lon * 1000, y: lat * 1000 }
}

const BUILDINGS: AuthorBuilding[] = [
  b('in1', 0.5, 0.5),
  b('in2', 0.7, 0.6, 'e1'),
  b('outside', 5.0, 5.0),
  b('stranded', 0.6, 0.55, null),
]

const BOX = { lon1: 0, lat1: 0, lon2: 1, lat2: 1 }

describe('box selection', () => {
  it('normalises a box drawn from any corner', () => {
    const a = normaliseBox({ lon1: 1, lat1: 1, lon2: 0, lat2: 0 })
    expect(a).toEqual({ minLon: 0, minLat: 0, maxLon: 1, maxLat: 1 })
  })

  it('selects by centroid, so a building is in or out as a whole', () => {
    expect(buildingsInBox(BUILDINGS, BOX).map((x) => x.id)).toEqual(['in1', 'in2'])
  })

  it('leaves out buildings with no road to spawn onto', () => {
    // A household cannot be placed on one, so counting it would make the number on
    // screen disagree with the package written.
    expect(buildingsInBox(BUILDINGS, BOX).map((x) => x.id)).not.toContain('stranded')
    expect(unspawnableInBox(BUILDINGS, BOX).map((x) => x.id)).toEqual(['stranded'])
  })

  it('excludes anything outside the box', () => {
    expect(buildingsInBox(BUILDINGS, BOX).map((x) => x.id)).not.toContain('outside')
  })
})

describe('household selection', () => {
  const picked = buildingsInBox(BUILDINGS, BOX)

  it('defaults one agent per building', () => {
    expect(addToSelection([], picked)).toEqual([
      { building_id: 'in1', count: 1 },
      { building_id: 'in2', count: 1 },
    ])
  })

  it('accumulates across boxes without duplicating', () => {
    const once = addToSelection([], picked)
    expect(addToSelection(once, picked)).toHaveLength(2)
  })

  it('keeps a count already edited when a later box overlaps', () => {
    const edited = setCount(addToSelection([], picked), 'in1', 6)
    const after = addToSelection(edited, picked)
    expect(after.find((h) => h.building_id === 'in1')?.count).toBe(6)
  })

  it('removes a box from the selection', () => {
    const after = removeFromSelection(addToSelection([], picked), [BUILDINGS[0]])
    expect(after.map((h) => h.building_id)).toEqual(['in2'])
  })

  it('toggles one building in and out', () => {
    const on = toggleBuilding([], BUILDINGS[0])
    expect(on).toHaveLength(1)
    expect(toggleBuilding(on, BUILDINGS[0])).toHaveLength(0)
  })

  it('refuses to select a building with no road', () => {
    expect(toggleBuilding([], BUILDINGS[3])).toHaveLength(0)
  })

  it('clamps a count to what the backend accepts', () => {
    const one = addToSelection([], [BUILDINGS[0]])
    expect(setCount(one, 'in1', 0)[0].count).toBe(1)
    expect(setCount(one, 'in1', -5)[0].count).toBe(1)
    expect(setCount(one, 'in1', 9999)[0].count).toBe(MAX_AGENTS_PER_BUILDING)
    expect(setCount(one, 'in1', 3.4)[0].count).toBe(3)
  })

  it('sets every count at once', () => {
    const all = setAllCounts(addToSelection([], picked), 4)
    expect(all.map((h) => h.count)).toEqual([4, 4])
  })

  it('counts buildings, agents, and distinct roads', () => {
    const index = new Map(BUILDINGS.map((x) => [x.id, x]))
    const households = setCount(addToSelection([], picked), 'in1', 3)
    expect(totals(households, index)).toEqual({ buildings: 2, agents: 4, roads: 2 })
  })
})

describe('alert area selection', () => {
  // Every area is a subset of the households. A building holding no agents cannot be
  // evacuated, and including it would pull its road into the ordered area, which would
  // order households that were never selected. A household belongs to at most one area,
  // so one colour on the map always answers which order reaches a building.
  const picked = buildingsInBox(BUILDINGS, BOX)
  const bothHouseholds = new Set(['in1', 'in2'])
  const onlyOne = new Set(['in1'])

  const area = (key: string, members: string[] = [], issueTimeS = 6300): DraftAlertArea => ({
    key,
    name: key,
    color: '#E69F00',
    wave: null,
    issueTimeS,
    hazardText: 'Go now.',
    comfortCentre: null,
    members,
  })
  const membersOf = (areas: DraftAlertArea[], key: string) =>
    areas.find((a) => a.key === key)?.members ?? []

  it('adds a box without duplicating members', () => {
    const once = assignAreaMembers([area('a')], 'a', picked, bothHouseholds)
    expect(membersOf(assignAreaMembers(once, 'a', picked, bothHouseholds), 'a')).toEqual(['in1', 'in2'])
  })

  it('skips buildings in the box that are not households', () => {
    expect(membersOf(assignAreaMembers([area('a')], 'a', picked, onlyOne), 'a')).toEqual(['in1'])
  })

  it('adds nothing when no household is selected', () => {
    expect(membersOf(assignAreaMembers([area('a')], 'a', picked, new Set()), 'a')).toEqual([])
  })

  it('moves a household out of the area that held it', () => {
    const before = [area('a', ['in1', 'in2']), area('b')]
    const after = assignAreaMembers(before, 'b', [BUILDINGS[0]], bothHouseholds)
    expect(membersOf(after, 'a')).toEqual(['in2'])
    expect(membersOf(after, 'b')).toEqual(['in1'])
  })

  it('removes a box from one area', () => {
    const after = clearAreaMembers([area('a', ['in1', 'in2'])], 'a', [BUILDINGS[0]])
    expect(membersOf(after, 'a')).toEqual(['in2'])
  })

  it('toggles one member', () => {
    expect(membersOf(toggleAreaMember([area('a', ['in1'])], 'a', 'in1', bothHouseholds), 'a')).toEqual([])
    expect(membersOf(toggleAreaMember([area('a')], 'a', 'in1', bothHouseholds), 'a')).toEqual(['in1'])
  })

  it('refuses to order a building that is not a household', () => {
    expect(membersOf(toggleAreaMember([area('a')], 'a', 'in2', onlyOne), 'a')).toEqual([])
  })

  it('drops members that stop being households', () => {
    expect(membersOf(pruneAreas([area('a', ['in1', 'in2'])], onlyOne), 'a')).toEqual(['in1'])
  })

  it('leaves the areas alone when every member is still a household', () => {
    const areas = [area('a', ['in1', 'in2'])]
    expect(pruneAreas(areas, bothHouseholds)).toBe(areas)
  })

  it('gives each new area a colour no other area is using', () => {
    const first = newArea([])
    const second = newArea([first])
    expect(second.color).not.toBe(first.color)
    expect(AREA_PALETTE).toContain(second.color)
  })

  // A new area carries no order, so adding one never writes a broadcast nobody chose.
  it('starts a new area with no order', () => {
    expect(newArea([]).issueTimeS).toBeNull()
    expect(newArea([]).wave).toBeNull()
  })

  it('separates areas that will broadcast from the ones still to be decided', () => {
    const areas = [area('a', ['in1'], 6300), { ...area('b', ['in2'], 0), issueTimeS: null }]
    expect(orderedAreas(areas).map((x) => x.key)).toEqual(['a'])
    expect(unassignedAreas(areas).map((x) => x.key)).toEqual(['b'])
  })
})

describe('record areas', () => {
  const payload = {
    package: 'p',
    households: 3,
    areas: [
      {
        name: 'westwood_hills', label: 'Westwood Hills', community: 'Westwood Hills',
        building_ids: ['in1'], agents: 1, edges: 1, ordered: true,
        wave: 'EA-1', issue_time_s: 6300, color: '#E69F00',
        hazard_text: 'Evacuate Westwood Hills.', comfort_centre: 'Black Point',
      },
      {
        name: 'stillwater_lake', label: 'Stillwater Lake', community: 'Stillwater Lake',
        building_ids: ['in2'], agents: 1, edges: 1, ordered: false,
        wave: null, issue_time_s: null, color: '#56B4E9',
      },
      {
        name: 'gone', label: 'Gone', community: 'Gone',
        building_ids: ['missing'], agents: 1, edges: 1, ordered: true,
        wave: 'EA-2', issue_time_s: 9660, color: '#F0E442',
      },
    ],
  }

  it('makes a household of every placed building and an area of every community', () => {
    const seeded = areasFromRecord(payload, BUILDINGS)
    expect(seeded.households.map((h) => h.building_id)).toEqual(['in1', 'in2'])
    expect(seeded.areas.map((a) => a.name)).toEqual(['westwood_hills', 'stillwater_lake'])
    expect(seeded.areas[0].issueTimeS).toBe(6300)
    expect(seeded.areas[0].comfortCentre).toBe('Black Point')
  })

  // A community no broadcast named still becomes an area, because deciding where its
  // households belong is a judgement the record cannot settle and the operator needs to
  // see them to make it. Carrying no time is what keeps it out of the written schedule.
  it('gives an unordered community an area with no issue time', () => {
    const stillwater = areasFromRecord(payload, BUILDINGS).areas.find(
      (a) => a.name === 'stillwater_lake',
    )
    expect(stillwater?.members).toEqual(['in2'])
    expect(stillwater?.issueTimeS).toBeNull()
    expect(stillwater?.wave).toBeNull()
  })

  it('sorts ordered areas before the ones still to be decided', () => {
    const seeded = areasFromRecord(payload, BUILDINGS)
    expect(seeded.areas[seeded.areas.length - 1].issueTimeS).toBeNull()
  })

  // An area holding households and no order writes nothing, so those households run
  // unordered instead of being swept into somebody else's broadcast.
  it('writes no broadcast for an area with no issue time', () => {
    const seeded = areasFromRecord(payload, BUILDINGS)
    const draft = buildDraft({
      id: 'authored', label: '', description: '', sourcePackage: 'src',
      households: seeded.households, areas: seeded.areas,
      areaChannel: 'broadcast', areaInstruction: 'evacuate_now',
      fires: [newFire(0, 1, 2)],
    })
    expect(draft.alert_areas?.map((a) => a.name)).toEqual(['westwood_hills'])
    expect(draft.alert_events).toHaveLength(1)
    expect(draft.households).toHaveLength(2)
  })

  it('counts buildings the map bundle does not hold', () => {
    expect(areasFromRecord(payload, BUILDINGS).skipped).toBe(1)
  })
})

describe('package naming', () => {
  it.each([
    ['', 'Give the package a name.'],
    ['ab', 'Use at least three characters.'],
    ['Caps', 'Start the name with a lowercase letter or a digit.'],
    ['has-dash', 'Use lowercase letters, digits, and underscores only.'],
    ['_lead', 'Start the name with a lowercase letter or a digit.'],
  ])('rejects %o', (id, message) => {
    expect(packageIdProblem(id)).toBe(message)
  })

  it('accepts what the backend accepts', () => {
    expect(packageIdProblem('halifax_authored_1')).toBeNull()
    expect(packageIdProblem('9lives')).toBeNull()
  })

  it('rejects a name longer than the backend allows', () => {
    expect(packageIdProblem('a'.repeat(65))).not.toBeNull()
  })

  it('suggests a name from a typed label', () => {
    expect(suggestPackageId('Westwood Ignition Area')).toBe('westwood_ignition_area')
    expect(suggestPackageId('  Three Towns / E0  ')).toBe('three_towns_e0')
  })
})

describe('reopening a package', () => {
  // Choosing a source package supplies the map and clears the canvas, which is right for
  // drawing something new. Reopening is the other case, where a package is adjusted and
  // saved under a new name.
  const stored = {
    package: 'p',
    households: [
      { building_id: 'in1', count: 3 },
      { building_id: 'in2', count: 1 },
      { building_id: 'missing', count: 1 },
    ],
    areas: [
      {
        name: 'ordered_area', label: 'ordered_area', color: '#CC79A7',
        building_ids: ['in1', 'missing'], wave: 'EA-2', issue_time_s: 9660,
        hazard_text: 'Go.', comfort_centre: null,
      },
      {
        name: 'edge_list_area', label: 'edge_list_area', color: '#009E73',
        building_ids: [], wave: 'EA-1', issue_time_s: 6300,
        hazard_text: '', comfort_centre: null,
      },
    ],
    fires: [],
    areas_without_buildings: ['edge_list_area'],
  }

  it('restores households with the counts the package recorded', () => {
    const opened = areasFromPackage(stored, BUILDINGS)
    expect(opened.households).toEqual([
      { building_id: 'in1', count: 3 },
      { building_id: 'in2', count: 1 },
    ])
    expect(opened.skipped).toBe(1)
  })

  it('restores an area with its wave, time and colour', () => {
    const opened = areasFromPackage(stored, BUILDINGS)
    expect(opened.areas).toHaveLength(1)
    expect(opened.areas[0].name).toBe('ordered_area')
    expect(opened.areas[0].wave).toBe('EA-2')
    expect(opened.areas[0].issueTimeS).toBe(9660)
    expect(opened.areas[0].color).toBe('#CC79A7')
  })

  // An area authored as an edge list carries no buildings the map can redraw, so it is
  // left out rather than restored as an empty area that would silently order nobody.
  it('drops an area whose buildings the map does not hold', () => {
    const opened = areasFromPackage(stored, BUILDINGS)
    expect(opened.areas.map((a) => a.name)).not.toContain('edge_list_area')
    expect(opened.areas[0].members).toEqual(['in1'])
  })

  it('survives a round trip back into a draft', () => {
    const opened = areasFromPackage(stored, BUILDINGS)
    const draft = buildDraft({
      id: 'reopened', label: '', description: '', sourcePackage: 'p',
      households: opened.households, areas: opened.areas,
      areaChannel: 'broadcast', areaInstruction: 'evacuate_now',
      fires: [newFire(0, 1, 2)],
    })
    expect(draft.alert_events?.[0].id).toBe('EA-2')
    expect(draft.alert_events?.[0].issue_time_s).toBe(9660)
    expect(draft.alert_areas?.[0].color).toBe('#CC79A7')
  })
})

describe('draft assembly', () => {
  const area = (name: string, members: string[], issueTimeS: number, wave: string | null = null) => ({
    key: name,
    name,
    color: '#E69F00',
    wave,
    issueTimeS,
    hazardText: 'Go now.',
    comfortCentre: null,
    members,
  })
  const base = {
    id: 'authored',
    label: 'Authored',
    description: '',
    sourcePackage: 'halifax_3town_e0',
    households: [{ building_id: 'in1', count: 2 }],
    areas: [] as DraftAlertArea[],
    areaChannel: 'broadcast',
    areaInstruction: 'evacuate_now',
    fires: [newFire(0, 100, 200)],
  }

  it('carries households and fires', () => {
    const draft = buildDraft(base)
    expect(draft.households).toEqual([{ building_id: 'in1', count: 2 }])
    expect(draft.fires).toHaveLength(1)
  })

  it('omits the alert schedule when no area holds a building', () => {
    expect(buildDraft(base).alert_areas).toBeUndefined()
    expect(buildDraft({ ...base, areas: [area('empty', [], 1800)] }).alert_events).toBeUndefined()
  })

  it('emits one area and one order when members are selected', () => {
    const draft = buildDraft({ ...base, areas: [area('ordered_area', ['in1', 'in2'], 1800)] })
    expect(draft.alert_areas?.[0].building_ids).toEqual(['in1', 'in2'])
    expect(draft.alert_events?.[0].issue_time_s).toBe(1800)
    expect(draft.alert_events?.[0].areas).toEqual(['ordered_area'])
  })

  it('carries each area colour into the package', () => {
    const one = { ...area('a', ['in1'], 1800), color: '#CC79A7' }
    expect(buildDraft({ ...base, areas: [one] }).alert_areas?.[0].color).toBe('#CC79A7')
  })

  // EA-3 named Haliburton Hills and Glen Arbour in one broadcast. Writing them as two
  // events at the same instant would claim two alerts where the record has one.
  it('merges areas sharing an issue time into one broadcast', () => {
    const draft = buildDraft({
      ...base,
      areas: [
        area('haliburton_hills', ['in1'], 15180, 'EA-3'),
        area('glen_arbour', ['in2'], 15180, 'EA-3'),
      ],
    })
    expect(draft.alert_events).toHaveLength(1)
    expect(draft.alert_events?.[0].id).toBe('EA-3')
    expect(draft.alert_events?.[0].areas).toEqual(['haliburton_hills', 'glen_arbour'])
  })

  it('numbers events in issue order when no wave is set', () => {
    const draft = buildDraft({
      ...base,
      areas: [area('late', ['in2'], 9660), area('early', ['in1'], 6300)],
    })
    expect(draft.alert_events?.map((e) => [e.id, e.issue_time_s])).toEqual([
      ['EA-1', 6300],
      ['EA-2', 9660],
    ])
  })

  it('omits an area that has no name', () => {
    expect(buildDraft({ ...base, areas: [area('  ', ['in1'], 1800)] }).alert_areas).toBeUndefined()
  })

  it('trims the text fields', () => {
    const draft = buildDraft({ ...base, id: '  authored  ', label: ' Authored ' })
    expect(draft.id).toBe('authored')
    expect(draft.label).toBe('Authored')
  })
})

describe('fire growth', () => {
  const fire = newFire(0, 10, 20)

  it('defaults to the shape the record’s own sources use', () => {
    expect(fire).toMatchObject({ t0: 0, r0: 30, growth_m_per_s: 0.3, max_r_m: 700 })
  })

  it('is nothing before it ignites', () => {
    expect(fireRadiusAt({ ...fire, t0: 500 }, 100)).toBe(0)
  })

  it('grows at the stated rate', () => {
    expect(fireRadiusAt({ ...fire, max_r_m: null }, 1000)).toBeCloseTo(330)
  })

  it('honours the radius cap', () => {
    expect(fireRadiusAt(fire, 100000)).toBe(700)
  })

  it('matches the simulator at the ignition instant', () => {
    expect(fireRadiusAt(fire, 0)).toBe(30)
  })
})

describe('authoring map style', () => {
  it('passes the MapLibre style specification', () => {
    expect(validateStyleMin(buildAuthorStyle())).toEqual([])
  })

  it('needs no network to render', () => {
    const style = buildAuthorStyle()
    expect(style.glyphs).toBeUndefined()
    expect(style.sprite).toBeUndefined()
    for (const source of Object.values(style.sources)) {
      expect(source.type).toBe('geojson')
    }
    for (const layer of style.layers) {
      expect((layer as { layout?: Record<string, unknown> }).layout?.['text-field']).toBeUndefined()
    }
  })

  it('keeps hazard red for fire alone', () => {
    const style = buildAuthorStyle()
    const red = style.layers.filter((layer) =>
      JSON.stringify((layer as { paint?: unknown }).paint ?? {}).includes('#D55E00'),
    )
    expect(red.every((layer) => layer.id.startsWith('fire-'))).toBe(true)
  })
})

describe('building geometry', () => {
  it('uses the footprint when the bundle carries one', () => {
    const withPoly = { ...b('p', 0, 0), poly: [[0, 0], [0, 1], [1, 1]] as [number, number][] }
    const feature = buildingFeature(withPoly, { household: true, area: false })
    const ring = (feature.geometry as GeoJSON.Polygon).coordinates[0]
    expect(ring).toHaveLength(4)
    expect(ring[0]).toEqual(ring[ring.length - 1])
  })

  it('falls back to a square when it does not', () => {
    const feature = buildingFeature(b('c', 0, 45), { household: false, area: false })
    expect((feature.geometry as GeoJSON.Polygon).coordinates[0]).toHaveLength(5)
  })

  it('marks a building with no road as stranded', () => {
    const feature = buildingFeature(b('s', 0, 0, null), { household: false, area: false })
    expect(feature.properties?.stranded).toBe(true)
  })

  it('closes a square ring', () => {
    const ring = squareRing(0, 45, 10)
    expect(ring[0]).toEqual(ring[ring.length - 1])
  })

  it('draws the dragged box as a closed rectangle from any corner', () => {
    const ring = (boxFeature(1, 1, 0, 0).geometry as GeoJSON.Polygon).coordinates[0]
    expect(ring).toHaveLength(5)
    expect(ring[0]).toEqual([0, 0])
    expect(ring[2]).toEqual([1, 1])
  })
})

describe('per-building agent counts', () => {
  const picked = buildingsInBox(BUILDINGS, BOX)

  it('raises one building without touching the others', () => {
    const after = setCount(addToSelection([], picked), 'in1', 20)
    expect(after.find((h) => h.building_id === 'in1')?.count).toBe(20)
    expect(after.find((h) => h.building_id === 'in2')?.count).toBe(1)
  })

  it('survives a later box that overlaps it', () => {
    const raised = setCount(addToSelection([], picked), 'in1', 20)
    expect(addToSelection(raised, picked).find((h) => h.building_id === 'in1')?.count).toBe(20)
  })

  it('is overwritten by the set-everything field, which is what that field says', () => {
    const raised = setCount(addToSelection([], picked), 'in1', 20)
    expect(setAllCounts(raised, 2).map((h) => h.count)).toEqual([2, 2])
  })

  it('ignores a building that is not a household', () => {
    const one = addToSelection([], [BUILDINGS[0]])
    expect(setCount(one, 'not-selected', 9)).toEqual(one)
  })

  it('feeds the agent total', () => {
    const index = new Map(BUILDINGS.map((x) => [x.id, x]))
    const raised = setCount(addToSelection([], picked), 'in1', 20)
    expect(totals(raised, index).agents).toBe(21)
  })

  it('is carried into the draft per building', () => {
    const raised = setCount(addToSelection([], picked), 'in1', 20)
    const draft = buildDraft({
      id: 'authored', label: '', description: '', sourcePackage: 'src',
      households: raised, areas: [],
      areaChannel: 'broadcast', areaInstruction: 'evacuate_now',
      fires: [newFire(0, 1, 2)],
    })
    expect(draft.households).toEqual([
      { building_id: 'in1', count: 20 },
      { building_id: 'in2', count: 1 },
    ])
  })
})

describe('building state for the map', () => {
  it('marks a raised building so it can be found again', () => {
    const feature = buildingFeature(b('big', 0, 0), { household: true, area: false, count: 20 })
    expect(feature.properties?.multi).toBe(true)
    expect(feature.properties?.count).toBe(20)
  })

  it('leaves an ordinary house unmarked', () => {
    const feature = buildingFeature(b('house', 0, 0), { household: true, area: false, count: 1 })
    expect(feature.properties?.multi).toBe(false)
  })

  it('carries both states, since an ordered building is always a household', () => {
    const feature = buildingFeature(b('both', 0, 0), { household: true, area: true })
    expect(feature.properties?.hh).toBe(true)
    expect(feature.properties?.area).toBe(true)
  })

  it('colours an ordered household differently from a merely selected one', () => {
    const fill = buildAuthorStyle().layers.find((l) => l.id === 'building-fill')
    const expression = JSON.stringify((fill as { paint?: unknown }).paint)
    // Area is tested before household, so an ordered household reads orange.
    expect(expression.indexOf('"area"')).toBeLessThan(expression.indexOf('"hh"'))
  })
})
