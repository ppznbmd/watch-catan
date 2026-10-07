// The board, drawn from the geometry the engine recorded at game_start.
//
// Catanatron uses cube coordinates on a pointy-top hex (see
// models/coordinate_system.py: neighbours are E, W, NE, NW, SE, SW; a tile's
// corners are N, NE, SE, S, SW, NW). That fixes the projection:
//
//     px = (sqrt(3)/2 R) (x - y)      py = 1.5 R z
//
// Verified against a real run before a line of this was drawn: all 96 nodes land
// on exactly one position each, and all 132 edges come out one side length long.
// Guessing the axis convention here produces a board that looks plausible and is
// wrong — tiles adjacent on screen that are not adjacent in the engine.

const R = 60;
const W2 = (Math.sqrt(3) / 2) * R;

const VERTEX = {
  NORTH: [0, -R],
  NORTHEAST: [W2, -R / 2],
  SOUTHEAST: [W2, R / 2],
  SOUTH: [0, R],
  SOUTHWEST: [-W2, R / 2],
  NORTHWEST: [-W2, -R / 2],
};

// Which two corners of a water tile a port faces, from map.py's
// PORT_DIRECTION_TO_NODEREFS. The dock lines are drawn to these.
const PORT_CORNERS = {
  WEST: ["NORTHWEST", "SOUTHWEST"],
  NORTHWEST: ["NORTH", "NORTHWEST"],
  NORTHEAST: ["NORTHEAST", "NORTH"],
  EAST: ["SOUTHEAST", "NORTHEAST"],
  SOUTHEAST: ["SOUTH", "SOUTHEAST"],
  SOUTHWEST: ["SOUTHWEST", "SOUTH"],
};

const SVG_NS = "http://www.w3.org/2000/svg";

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function center([x, y, z]) {
  return [W2 * (x - y), 1.5 * R * z];
}

function corners([x, y, z]) {
  const [cx, cy] = center([x, y, z]);
  return Object.values(VERTEX).map(([dx, dy]) => `${cx + dx},${cy + dy}`).join(" ");
}

// How many of the 36 dice outcomes hit this number. Catan prints it as pips, and
// it is the single most important thing about a tile.
function pips(number) {
  return number ? 6 - Math.abs(7 - number) : 0;
}

export class Board {
  constructor(svg) {
    this.svg = svg;
    this.nodes = new Map(); // node id -> [x, y]
  }

  // The half that never changes: water, ports, tiles, numbers.
  drawStatic(geometry) {
    this.svg.replaceChildren();
    this.nodes.clear();
    for (const [id, node] of Object.entries(geometry.nodes)) {
      const [cx, cy] = center(node.tile_coordinate);
      const [dx, dy] = VERTEX[node.direction];
      this.nodes.set(Number(id), [cx + dx, cy + dy]);
    }

    const water = el("g", { class: "layer-water" }, this.svg);
    const land = el("g", { class: "layer-land" }, this.svg);
    this.dynamic = el("g", { class: "layer-play" }, this.svg);

    let minX = 0, maxX = 0, minY = 0, maxY = 0;
    for (const { coordinate, tile } of geometry.tiles) {
      const [cx, cy] = center(coordinate);
      minX = Math.min(minX, cx - W2); maxX = Math.max(maxX, cx + W2);
      minY = Math.min(minY, cy - R); maxY = Math.max(maxY, cy + R);

      if (tile.type === "WATER" || tile.type === "PORT") {
        el("polygon", { points: corners(coordinate), class: "hex hex-water" }, water);
        if (tile.type === "PORT") this.drawPort(water, coordinate, tile);
        continue;
      }
      const kind = (tile.type === "DESERT" ? "DESERT" : tile.resource).toLowerCase();
      el("polygon", {
        points: corners(coordinate),
        class: `hex hex-${kind}`,
      }, land);

      // What the tile produces, as a picture. It sits above centre so the number
      // token can sit below it without covering it — the two halves of a Catan
      // tile you actually read: what it makes, and how often.
      this.drawGlyph(land, cx, cy - R * 0.30, kind, R * 0.86);
      if (tile.number) this.drawNumber(land, cx, cy + R * 0.34, tile.number);
    }

    const pad = 14;
    this.svg.setAttribute("viewBox",
      `${minX - pad} ${minY - pad} ${maxX - minX + pad * 2} ${maxY - minY + pad * 2}`);
  }

  drawGlyph(parent, cx, cy, name, size) {
    el("use", {
      href: `#i-${name}`,
      x: cx - size / 2, y: cy - size / 2, width: size, height: size,
      class: `glyph glyph-${name}`,
    }, parent);
  }

  drawNumber(parent, cx, cy, number) {
    const hot = number === 6 || number === 8;  // the two everyone builds on
    const g = el("g", { class: `token${hot ? " token-hot" : ""}` }, parent);
    el("circle", { cx, cy, r: R * 0.285 }, g);
    const text = el("text", { x: cx, y: cy - R * 0.02, class: "token-number" }, g);
    text.textContent = number;
    const count = pips(number);
    const width = R * 0.068;
    for (let i = 0; i < count; i++) {
      el("circle", {
        cx: cx + (i - (count - 1) / 2) * width * 2.1,
        cy: cy + R * 0.155,
        r: width * 0.58,
        class: "pip",
      }, g);
    }
  }

  drawPort(parent, coordinate, tile) {
    const [cx, cy] = center(coordinate);
    const [a, b] = PORT_CORNERS[tile.direction] || [];
    if (!a) return;
    const g = el("g", { class: "port" }, parent);
    for (const which of [a, b]) {
      const [dx, dy] = VERTEX[which];
      el("line", { x1: cx, y1: cy, x2: cx + dx, y2: cy + dy, class: "dock" }, g);
    }
    // Pull the sign toward the land, or it sits in open water far from its docks.
    const [ax, ay] = VERTEX[a], [bx, by] = VERTEX[b];
    const lx = cx + (ax + bx) / 2 * 0.52, ly = cy + (ay + by) / 2 * 0.52;
    if (tile.resource) {
      this.drawGlyph(g, lx, ly - R * 0.16, tile.resource.toLowerCase(), R * 0.46);
    }
    const label = el("text", {
      x: lx, y: ly + (tile.resource ? R * 0.28 : 0), class: "port-label",
    }, g);
    label.textContent = tile.resource ? "2:1" : "3:1";
  }

  // The half that changes every ply.
  draw(frame, previous) {
    if (!this.dynamic) return;
    this.dynamic.replaceChildren();

    const fresh = changed(frame, previous);

    const roads = [];
    for (const [key, color] of Object.entries(frame.roads || {})) {
      const [a, b] = JSON.parse(key);
      const pa = this.nodes.get(a), pb = this.nodes.get(b);
      if (!pa || !pb) continue;
      roads.push([key, color, { x1: pa[0], y1: pa[1], x2: pb[0], y2: pb[1] }]);
    }
    // Three passes over every road, not three strokes per road in turn: a later
    // road's halo painted over an earlier road's body bites a wedge out of a
    // road that is really there.
    //
    // A piece needs two different separations and one stroke cannot give both.
    // The paper bed lifts it off the tile it sits on; the dark edge divides it
    // from the pieces it touches, which at a node are its own roads in its own
    // colour. With only the bed, a settlement at the end of a road merged into
    // it and the pair read as a single funnel.
    for (const [, , attrs] of roads) {
      el("line", { ...attrs, class: "road-bed" }, this.dynamic);
    }
    for (const [, , attrs] of roads) {
      el("line", { ...attrs, class: "road-edge" }, this.dynamic);
    }
    for (const [key, color, attrs] of roads) {
      el("line", {
        ...attrs,
        class: `road seat-${color}${fresh.roads.has(key) ? " is-new" : ""}`,
      }, this.dynamic);
    }

    for (const [id, [color, type]] of Object.entries(frame.buildings || {})) {
      const point = this.nodes.get(Number(id));
      if (!point) continue;
      this.drawBuilding(point, color, type, fresh.buildings.has(id));
    }

    if (frame.robber) this.drawRobber(frame.robber);
  }

  drawBuilding([x, y], color, type, isNew) {
    // The two pieces are told apart by their roofline, not by their size. Both
    // were houses before, one slightly larger, and at 15px on a coloured tile
    // that is no difference at all: a pitched roof reads as a village, a flat
    // top with a tower beside it reads as a city, at any size.
    const shape = type === "CITY" ? this.city(x, y) : this.settlement(x, y);
    el("polygon", { points: shape, class: "building-bed" }, this.dynamic);
    el("polygon", {
      points: shape,
      class: `building building-${type.toLowerCase()} seat-${color}${isNew ? " is-new" : ""}`,
    }, this.dynamic);
  }

  settlement(x, y) {
    const s = R * 0.23;
    return `${x - s},${y + s} ${x - s},${y - s * 0.25} ${x},${y - s * 1.35}
            ${x + s},${y - s * 0.25} ${x + s},${y + s}`;
  }

  city(x, y) {
    // A tall narrow tower beside a long low hall, flat-topped throughout. Making
    // the two volumes differ in both dimensions is what carries the silhouette:
    // when they were closer in size it read as one block with a notch in it.
    const u = R * 0.15;
    return `${x - 2.3 * u},${y + 1.5 * u} ${x - 2.3 * u},${y - 2.7 * u}
            ${x - 0.6 * u},${y - 2.7 * u} ${x - 0.6 * u},${y - 0.6 * u}
            ${x + 2.3 * u},${y - 0.6 * u} ${x + 2.3 * u},${y + 1.5 * u}`;
  }

  drawRobber(coordinate) {
    const [cx, cy] = center(coordinate);
    const size = R * 0.66;
    // It stands on the tile art, where a real one does. The halo behind it is
    // what keeps it legible over five different tile colours.
    el("circle", { cx, cy: cy - R * 0.26, r: size * 0.56, class: "robber-halo" },
       this.dynamic);
    el("use", {
      href: "#i-robber",
      x: cx - size / 2, y: cy - R * 0.26 - size / 2, width: size, height: size,
      class: "robber",
    }, this.dynamic);
  }
}

// What appeared since the previous position. Drawing attention to it is the
// difference between watching a game and watching a board redraw itself.
function changed(frame, previous) {
  const roads = new Set(), buildings = new Set();
  if (!previous) return { roads, buildings };
  for (const key of Object.keys(frame.roads || {})) {
    if (!(key in (previous.roads || {}))) roads.add(key);
  }
  for (const [id, value] of Object.entries(frame.buildings || {})) {
    const before = (previous.buildings || {})[id];
    if (!before || before[1] !== value[1]) buildings.add(id);  // built, or upgraded
  }
  return { roads, buildings };
}
