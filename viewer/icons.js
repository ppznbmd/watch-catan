// The game's furniture: one sprite sheet, drawn once into the document, used by
// the board and by the panels alike.
//
// Everything here is inline SVG on a 24x24 grid. No image files and no icon font:
// the viewer has to work from a directory served by the stdlib, offline, with no
// build step, and an emoji would not survive being scaled onto a hex.
//
// Shapes are silhouettes on purpose. They sit under a number token and behind a
// road, so they have to read at a glance and then get out of the way.

const SHAPES = {
  // Forest.
  wood: `<path d="M12 2.5 18 11H6Z"/><path d="M12 7.5 19.5 17H4.5Z"/>
         <rect x="10.6" y="16" width="2.8" height="5.5" rx="0.8"/>`,
  // Hills: fired clay, stacked.
  brick: `<rect x="2.5" y="13.5" width="8.6" height="5.6" rx="1"/>
          <rect x="12.9" y="13.5" width="8.6" height="5.6" rx="1"/>
          <rect x="7.7" y="7" width="8.6" height="5.6" rx="1"/>`,
  // Pasture. The head is held clear of the fleece: overlapping them merges the
  // two into one lump that reads as a cloud, or a tree.
  sheep: `<circle cx="11.4" cy="10.6" r="3.5"/><circle cx="15.8" cy="10" r="3.6"/>
          <circle cx="19" cy="12.6" r="3"/>
          <ellipse cx="15.2" cy="14" rx="6.2" ry="4.6"/>
          <rect x="11.4" y="17.6" width="1.9" height="4" rx="0.9"/>
          <rect x="17" y="17.6" width="1.9" height="4" rx="0.9"/>
          <ellipse cx="6.2" cy="9.6" rx="1.5" ry="2.2" transform="rotate(-28 6.2 9.6)"/>
          <circle cx="6.2" cy="13.2" r="3.1"/>
          <ellipse cx="3.1" cy="14.8" rx="1.9" ry="1.4"/>`,
  // Fields. One ear, drawn properly: paired grains up a spine. Three vague
  // sprays at this size came out looking like tulips.
  wheat: `<path d="M11.2 22.5V12h1.6v10.5Z"/>
          <ellipse cx="12" cy="3.6" rx="1.5" ry="2.6"/>
          <ellipse cx="9.7" cy="6.6" rx="1.5" ry="2.6" transform="rotate(-32 9.7 6.6)"/>
          <ellipse cx="14.3" cy="6.6" rx="1.5" ry="2.6" transform="rotate(32 14.3 6.6)"/>
          <ellipse cx="9.2" cy="10.2" rx="1.5" ry="2.6" transform="rotate(-32 9.2 10.2)"/>
          <ellipse cx="14.8" cy="10.2" rx="1.5" ry="2.6" transform="rotate(32 14.8 10.2)"/>
          <ellipse cx="8.8" cy="13.8" rx="1.5" ry="2.6" transform="rotate(-32 8.8 13.8)"/>
          <ellipse cx="15.2" cy="13.8" rx="1.5" ry="2.6" transform="rotate(32 15.2 13.8)"/>`,
  // Mountains — the Catan ore tile. A faceted crystal read as a shield, and a
  // shield is already the knight.
  ore: `<path d="M0.8 20.8 7.6 6.5 12 14.2 15.6 8.6 23.2 20.8Z"/>
        <path d="M7.6 6.5 5 12l1.6-.7 1.5.9 1.6-1.3Z" opacity="0.45"/>`,
  desert: `<circle cx="17.8" cy="6.4" r="3"/>
           <path d="M1.5 18.5q5-6.5 10 0 5-6.5 11 0v4h-21Z"/>`,

  // The pieces.
  robber: `<circle cx="12" cy="6.2" r="3.9"/>
           <path d="M5.2 21.5c0-6.6 3-10.8 6.8-10.8s6.8 4.2 6.8 10.8Z"/>`,
  knight: `<path d="M12 2 21 5v6.2c0 5.4-3.9 9.4-9 10.8-5.1-1.4-9-5.4-9-10.8V5Z"/>`,
  card: `<path fill-rule="evenodd" d="M5 2.5h14a2.5 2.5 0 0 1 2.5 2.5v14a2.5 2.5 0
           0 1-2.5 2.5H5a2.5 2.5 0 0 1-2.5-2.5V5A2.5 2.5 0 0 1 5 2.5Z
           M7 7.6h10v1.7H7Zm0 3.7h10V13H7Zm0 3.7h6.5v1.7H7Z"/>`,
  // A road running away from you, centre line and all.
  road: `<path fill-rule="evenodd" d="M7.4 22 10.9 2.5h2.2L16.6 22Z
           M11.5 18.3h1.3v2.6h-1.3Zm.25-4.6h1.1v2.3h-1.1Zm.25-4.3h.9v2.1h-.9Z
           M12.2 5.2h.7v1.8h-.7Z"/>`,
};

export const SPRITE = `<svg class="sprite" aria-hidden="true">
  <defs>${Object.entries(SHAPES).map(([name, body]) =>
    `<symbol id="i-${name}" viewBox="0 0 24 24">${body}</symbol>`).join("")}</defs>
</svg>`;

export function icon(name, cls = "") {
  return `<svg class="icon ${cls}" viewBox="0 0 24 24" aria-hidden="true">
            <use href="#i-${name}"/></svg>`;
}

// Where the pips sit on each face. Written out rather than computed: a die is a
// fixed piece of design, not a grid.
const PIPS = {
  1: [[12, 12]],
  2: [[7.4, 7.4], [16.6, 16.6]],
  3: [[7.4, 7.4], [12, 12], [16.6, 16.6]],
  4: [[7.4, 7.4], [16.6, 7.4], [7.4, 16.6], [16.6, 16.6]],
  5: [[7.4, 7.4], [16.6, 7.4], [12, 12], [7.4, 16.6], [16.6, 16.6]],
  6: [[7.4, 6.4], [16.6, 6.4], [7.4, 12], [16.6, 12], [7.4, 17.6], [16.6, 17.6]],
};

export function die(face, cls = "") {
  const pips = (PIPS[face] || []).map(([x, y]) =>
    `<circle cx="${x}" cy="${y}" r="2.05"/>`).join("");
  return `<svg class="die ${cls}" viewBox="0 0 24 24" role="img" aria-label="${face}">
            <rect class="die-face" x="1.4" y="1.4" width="21.2" height="21.2" rx="4.6"/>
            <g class="die-pips">${pips}</g></svg>`;
}
