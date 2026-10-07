// The spectator: board, hands, and what was said.
//
// Following a live match and replaying a finished one are the same code path.
// `/api/run?from=N` answers with whatever is new, so "live" is nothing more than
// asking again on a timer and, if the viewer is already at the end, moving on.

import { Board } from "./board.js";
import { SPRITE, icon, die } from "./icons.js";

const RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"];
const DEV_CARDS = ["KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"];
const POLL_MS = 1000;

const $ = (id) => document.getElementById(id);

// One sprite sheet for the whole document: the board's <use> elements and the
// panels' inline icons both point at it.
document.body.insertAdjacentHTML("afterbegin", SPRITE);

const board = new Board($("board"));

const view = {
  path: null,
  meta: null,
  frames: [],
  summary: null,
  complete: false,
  index: 0,
  playing: false,
  following: true,   // at the end, and staying there as new frames land
  timer: null,
  poller: null,
  rendered: -1,      // how much of the feed is already in the DOM
};

// ------------------------------------------------------------------ loading

// The picker is rebuilt on a timer, so a match started after the page was opened
// shows up, and one that ends stops claiming to be live.
const RUNS_MS = 10000;
let runs = [];

async function loadRuns() {
  ({ runs } = await (await fetch("/api/runs")).json());
  const picker = $("run-picker");
  const selected = picker.value;
  picker.replaceChildren();
  if (!runs.length) {
    picker.append(new Option("no runs on disk", ""));
    return;
  }
  const groups = new Map([["live", "playing now"], ["real", "real matches"], ["dry", "dry runs"]]);
  for (const [key, label] of groups) {
    const group = document.createElement("optgroup");
    group.label = label;
    groups.set(key, group);
  }
  for (const run of runs) {
    const option = new Option(describe(run), run.path);
    option.title = run.path;
    groups.get(run.status === "live" ? "live" : run.scratch ? "dry" : "real").append(option);
  }
  for (const group of groups.values()) if (group.children.length) picker.append(group);
  if (selected) picker.value = selected;
  markLive();
  if (view.path) $("live").hidden = view.complete || statusOf(view.path) !== "live";
  return runs[0].path;
}

function markLive() {
  const picker = $("run-picker");
  picker.classList.toggle("is-live", statusOf(picker.value) === "live");
}

function statusOf(path) {
  return runs.find((run) => run.path === path)?.status;
}

// What a run is, in the words someone choosing between them needs: whether it is
// happening now, and if not, how long ago it happened.
function describe(run) {
  const when = run.started ? new Date(run.started * 1000) : null;
  if (!when) return run.name;
  if (run.status === "live") {
    // Live runs share one group, so a dry one has to say so itself.
    const dry = run.scratch ? " (dry run)" : "";
    return `\u25CF LIVE${dry} \u00B7 started ${clock(when)}, ${ago(when)}`;
  }
  const end = run.status === "finished" ? "finished" : "stopped mid-game";
  return `${day(when)} ${clock(when)} \u00B7 ${ago(when)} \u00B7 ${end}`;
}

const clock = (d) => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

function day(d) {
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  if (d.toDateString() === today.toDateString()) return "today";
  if (d.toDateString() === yesterday.toDateString()) return "yesterday";
  return d.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" });
}

function ago(d) {
  const minutes = Math.round((Date.now() - d) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours} h ago`;
  return `${Math.round(hours / 24)} days ago`;
}

// A position is worth linking to: `?run=<path>&at=<ply>` reopens exactly this
// board, which is how you point someone at the trade you are arguing about.
function syncUrl() {
  const url = new URL(location.href);
  url.searchParams.set("run", view.path);
  url.searchParams.set("at", view.index);
  history.replaceState(null, "", url);
}

async function open(path, at = null) {
  stop();
  clearInterval(view.poller);
  Object.assign(view, {
    path, meta: null, frames: [], summary: null, complete: false,
    index: 0, following: true, rendered: -1,
  });
  $("feed").replaceChildren();
  $("run-name").textContent = path;

  const payload = await fetchFrames(0);
  if (payload.error) return fail(payload.error);

  view.meta = payload.meta;
  board.drawStatic(view.meta.board);
  $("empty").hidden = true;
  drawSeats();
  show(at === null ? view.frames.length - 1 : at);

  if (!view.complete) view.poller = setInterval(poll, POLL_MS);
}

async function fetchFrames(from) {
  const url = `/api/run?path=${encodeURIComponent(view.path)}&from=${from}`;
  const payload = await (await fetch(url)).json();
  if (payload.error) return payload;
  view.frames.push(...payload.frames);
  view.complete = payload.complete;
  view.summary = payload.summary;
  $("total").textContent = view.frames.length - 1;
  $("scrub").max = Math.max(0, view.frames.length - 1);
  // Not reaching `game_over` alone does not make a match live: its process may
  // have died. The listing knows which files are still being written.
  $("live").hidden = view.complete || statusOf(view.path) !== "live";
  return payload;
}

async function poll() {
  const before = view.frames.length;
  await fetchFrames(before);
  if (view.frames.length === before) return;
  if (view.following) show(view.frames.length - 1);
  else $("to-live").hidden = false;
  if (view.complete) {
    clearInterval(view.poller);
    $("to-live").hidden = true;
  }
}

function fail(message) {
  $("empty").hidden = false;
  $("empty").textContent = message;
}

// ------------------------------------------------------------------ rendering

function show(index) {
  if (!view.frames.length) return;
  index = Math.max(0, Math.min(index, view.frames.length - 1));
  const frame = view.frames[index];
  view.index = index;
  view.following = index === view.frames.length - 1;
  if (view.following) $("to-live").hidden = true;

  board.draw(frame, view.frames[index - 1]);
  $("turn").textContent = frame.turn;
  $("prompt").textContent = frame.prompt.replaceAll("_", " ").toLowerCase();
  $("position").textContent = index;
  $("scrub").value = index;
  showDice(frame);
  drawSeats();
  fillFeed();
}

function showDice(frame) {
  // The roll stays on screen until the next one: a spectator reads the board
  // against the number that produced it, not against the ply it is on.
  const roll = lastRoll(view.index);
  const box = $("dice");
  box.hidden = !roll;
  if (!roll) return;
  const total = roll[0] + roll[1];
  box.innerHTML = `${die(roll[0])}${die(roll[1])}<span class="total">${total}</span>`;
  box.classList.toggle("seven", total === 7);
}

function lastRoll(index) {
  for (let i = index; i >= 0; i--) {
    const action = view.frames[i].last_action;
    if (action && action.type === "ROLL") return action.result;
  }
  return null;
}

function drawSeats() {
  const frame = view.frames[view.index];
  const degraded = degradedSoFar();
  const seats = $("seats");
  seats.replaceChildren();

  // Around the table, not down a list: seats are laid out in the engine's turn
  // order — clockwise from the top-left corner — so the screen matches the order
  // play actually moves in. `meta.seats` is construction order, which is not it.
  const byColor = Object.fromEntries(view.meta.seats.map((s) => [s.color, s]));
  const order = view.meta.order.map((color) => byColor[color]).filter(Boolean);

  for (const [position, seat] of order.entries()) {
    const state = frame.players[seat.color] || {};
    const node = document.createElement("article");
    node.className = `seat seat-bg-${seat.color} seat-at-${position}`;
    node.classList.toggle("is-turn", frame.current_color === seat.color);
    // Only once the replay has actually reached the end: marking the winner while
    // scrubbing through turn 20 gives away the thing you are watching to find out.
    const atEnd = view.index === view.frames.length - 1;
    if (atEnd && view.summary?.winner === seat.color) node.classList.add("is-winner");

    const cards = DEV_CARDS.reduce((n, c) => n + (state.cards?.[c] || 0), 0);
    const knights = state.played?.KNIGHT || 0;
    const broken = degraded[seat.color] || 0;
    // Victory-point cards stay in the hand, face down: the total counts them, the
    // other players do not see them. Showing both is how a surprise win reads.
    const hiddenVp = state.cards?.VICTORY_POINT || 0;
    const vpTitle = hiddenVp
      ? `victory points: ${state.public_victory_points ?? 0} on the board, ${hiddenVp} from hidden dev cards`
      : "victory points";

    node.innerHTML = `
      <header>
        <span class="chip"></span>
        <span class="who">${escape(seat.name)}</span>
        <span class="vp" title="${vpTitle}">${state.victory_points ?? 0}${hiddenVp
          ? `<small>${hiddenVp} dev</small>` : ""}</span>
      </header>
      <p class="model">${escape(seat.model || "baseline bot")}</p>
      <ul class="hand">
        ${RESOURCES.map((r) => {
          const held = state.hand?.[r] || 0;
          return `<li class="res res-${r.toLowerCase()}${held ? "" : " zero"}"
                      title="${r.toLowerCase()}">
                    ${icon(r.toLowerCase())}<span>${held}</span>
                  </li>`;
        }).join("")}
      </ul>
      <p class="counters">
        <span${cards ? "" : ' class="zero"'}>${icon("card")}${cards}</span>
        <span${knights ? "" : ' class="zero"'} title="knights played">${icon("knight")}${knights}</span>
        <span${state.longest_road ? "" : ' class="zero"'} title="longest road">${icon("road")}${state.longest_road ?? 0}</span>
        ${state.has_road ? '<span class="badge">longest road</span>' : ""}
        ${state.has_army ? '<span class="badge">largest army</span>' : ""}
        ${broken ? `<span class="badge warn" title="replies that could not be used, or fell back to the cheap policy">${broken} degraded</span>` : ""}
      </p>`;
    seats.append(node);
  }
}

// A model that fails often still produces a complete, plausible match, because
// the reflex policy fills every gap. If this is not on screen, the viewer shows a
// persona that was not actually playing.
function degradedSoFar() {
  const tally = {};
  for (let i = 0; i <= view.index; i++) {
    for (const event of view.frames[i].degraded || []) {
      tally[event.color] = (tally[event.color] || 0) + 1;
    }
  }
  return tally;
}

// ------------------------------------------------------------------ the feed

function fillFeed() {
  const feed = $("feed");
  if (view.index < view.rendered) {          // scrubbed backwards: rebuild
    feed.replaceChildren();
    view.rendered = -1;
  }
  for (let i = view.rendered + 1; i <= view.index; i++) {
    for (const entry of entriesFor(view.frames[i])) feed.append(entry);
  }
  view.rendered = view.index;
  applyFilters();
  feed.scrollTop = feed.scrollHeight;  // scrollIntoView would move the page, not the list
}

// One ply, one story. Speech wins when there is any: the utterance already names
// the action it came with, so rendering the move as well would say it twice.
function entriesFor(frame) {
  const skipped = frame.events.find((e) => e.kind === "own_offer_skipped");
  if (skipped) return [selfOffer(skipped)];
  const passed = frame.events.find((e) => e.kind === "not_addressed");
  if (passed) return [notAddressed(passed)];
  if (frame.talk.length) return frame.talk.map(speech);
  return frame.last_action ? [move(frame.last_action)] : [];
}

// Catanatron polls the proposer about its own trade (it excludes whoever last
// answered, not whoever offered). Rendered as a plain rejection it reads as a
// player turning down its own offer, which never happened — the engine asked and
// we answered for it, without a model call.
function selfOffer(event) {
  const li = document.createElement("li");
  li.className = "entry move quirk";
  li.innerHTML = `<span class="chip seat-bg-${event.color}"></span>
    <span class="name">${escape(event.color)}</span>
    <span class="act">asked about its own offer — answered free</span>`;
  return li;
}

// A counter-offer taken up is put to one player, but the engine still walks the
// table. The others are answered for, without a model call; shown as a plain
// rejection it would read as a refusal nobody made.
function notAddressed(event) {
  const li = document.createElement("li");
  li.className = "entry move quirk";
  li.innerHTML = `<span class="chip seat-bg-${event.color}"></span>
    <span class="name">${escape(event.color)}</span>
    <span class="act">not party to that offer — answered free</span>`;
  return li;
}

function speech(utterance) {
  const li = document.createElement("li");
  li.className = `entry speech kind-${utterance.kind}`;
  const headline = utterance.kind === "pitch"
    ? `offers${utterance.to ? ` ${escape(utterance.to)} alone` : ""} ${deck(utterance.give)} for ${deck(utterance.want)}`
    : utterance.kind === "counter"
      ? `counters: would give ${deck(utterance.give)} for ${deck(utterance.want)}`
      : escape(utterance.described || utterance.kind);
  li.innerHTML = `
    <p class="who"><span class="chip seat-bg-${utterance.color}"></span>
      <span class="name">${escape(utterance.color)}</span>
      <span class="act">${headline}</span></p>
    ${utterance.text ? `<blockquote>${escape(utterance.text)}</blockquote>` : ""}
    ${utterance.reasoning ? `<p class="why">${escape(utterance.reasoning)}</p>` : ""}`;
  return li;
}

function move(action) {
  const li = document.createElement("li");
  li.className = "entry move";
  li.innerHTML = `<span class="chip seat-bg-${action.color}"></span>
    <span class="name">${escape(action.color)}</span>
    <span class="act">${escape(action.described)}</span>
    ${outcome(action)}`;
  return li;
}

// What the engine did in response — the half no decision event can know.
function outcome(action) {
  if (action.type === "ROLL" && Array.isArray(action.result)) {
    const total = action.result[0] + action.result[1];
    return `<span class="outcome rolled${total === 7 ? " seven" : ""}">
              ${die(action.result[0])}${die(action.result[1])}
              <span class="total">${total}</span></span>`;
  }
  if (action.type === "MOVE_ROBBER" && action.result) {
    return `<span class="outcome">took ${escape(action.result)}</span>`;
  }
  if (action.type === "BUY_DEVELOPMENT_CARD" && action.result) {
    return `<span class="outcome">${escape(action.result.replaceAll("_", " ").toLowerCase())}</span>`;
  }
  return "";
}

// A trade is read as cards at a table, not as a sentence. The count leads because
// that is what is being haggled over.
function deck(freqdeck) {
  const parts = (freqdeck || []).map((n, i) => {
    if (!n) return null;
    const name = RESOURCES[i].toLowerCase();
    return `<span class="card res-${name}" title="${n} ${name}">${n}${icon(name)}</span>`;
  }).filter(Boolean);
  return parts.length ? parts.join("") : "nothing";
}

function applyFilters() {
  document.body.classList.toggle("hide-moves", !$("show-moves").checked);
  document.body.classList.toggle("hide-why", !$("show-why").checked);
}

function escape(value) {
  return String(value ?? "").replace(/[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// ------------------------------------------------------------------ transport

function play() {
  if (view.playing) return stop();
  view.playing = true;
  $("play").innerHTML = "&#10073;&#10073;";
  tick();
}

function tick() {
  clearTimeout(view.timer);
  if (!view.playing) return;
  if (view.index >= view.frames.length - 1) {
    // At the end of a live match, wait for more rather than stopping.
    if (view.complete) return stop();
    view.timer = setTimeout(tick, POLL_MS);
    return;
  }
  show(view.index + 1);
  view.timer = setTimeout(tick, Number($("speed").value));
}

function stop() {
  view.playing = false;
  clearTimeout(view.timer);
  $("play").innerHTML = "&#9654;";
}

// ------------------------------------------------------------------ wiring

function goto(index) { stop(); show(index); syncUrl(); }

$("play").onclick = play;
$("back").onclick = () => goto(view.index - 1);
$("next").onclick = () => goto(view.index + 1);
$("scrub").oninput = (e) => goto(Number(e.target.value));
$("speed").onchange = () => { if (view.playing) tick(); };
$("show-moves").onchange = applyFilters;
$("show-why").onchange = applyFilters;
$("to-live").onclick = () => goto(view.frames.length - 1);
$("run-picker").onchange = (e) => { markLive(); e.target.value && open(e.target.value); };

$("theme").onclick = () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch { /* private window */ }
};
// `?theme=` wins over the stored choice, which wins over the system setting.
const asked = new URLSearchParams(location.search).get("theme");
if (asked === "light" || asked === "dark") {
  document.documentElement.dataset.theme = asked;
} else {
  try {
    const saved = localStorage.getItem("theme");
    if (saved) document.documentElement.dataset.theme = saved;
  } catch { /* private window */ }
}

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, select")) return;
  const keys = {
    " ": play,
    ArrowLeft: () => goto(view.index - 1),
    ArrowRight: () => goto(view.index + 1),
    Home: () => goto(0),
    End: () => goto(view.frames.length - 1),
  };
  if (keys[e.key]) { e.preventDefault(); keys[e.key](); }
});

const params = new URLSearchParams(location.search);
const newest = await loadRuns();
setInterval(() => document.activeElement !== $("run-picker") && loadRuns(), RUNS_MS);
const wanted = params.get("run") || newest;
if (wanted) {
  $("run-picker").value = wanted;
  markLive();
  const at = params.get("at");
  await open(wanted, at === null ? null : Number(at));
}
