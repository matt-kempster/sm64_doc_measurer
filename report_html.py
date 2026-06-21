"""Render a self-contained current-state HTML report.

One static file: the result JSON is embedded and a little vanilla JS renders it,
so it drops straight onto GitHub Pages with no external assets. The page has a
dark/light toggle, three headline scores (completeness, uniformity, wired-up),
per-kind bars for the two name-quality axes, a completeness-by-family table, a
semantic cross-reference table, and one sortable/filterable worklist.

The page is built to stay short: each section is a collapsible panel showing a
headline stat when closed, the long family table collapses by default and hides
fully-complete (100%) families, and every metric is *clickable* — a bar, a family
row, an entity, or a scorecard — to filter the worklist to it and jump there.

Substitution is by token (``__DATA__`` / ``__LABEL__``) rather than ``str.format``
so the CSS/JS braces don't need escaping.
"""

import html
import json

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SM64 Documentation — Current State</title>
<style>
  /* Base theme is dark; OS preference flips first paint; the toggle overrides. */
  :root {
    --bg: #0f1117; --panel: #161922; --panel2: #1c202b; --fg: #e6e8ee;
    --muted: #8b90a0; --line: #262a36; --accent: #58a6ff;
    --good: #3fb950; --mal: #d29922; --und: #f85149; --conv: #58a6ff; --sem: #a371f7;
    --asset: #db6d28;
    --shadow: 0 1px 0 rgba(0,0,0,.3);
  }
  @media (prefers-color-scheme: light) {
    :root:not([data-theme]) {
      --bg: #ffffff; --panel: #f6f8fa; --panel2: #eef1f5; --fg: #1f2328;
      --muted: #636c76; --line: #d0d7de; --accent: #0969da;
      --good: #1a7f37; --mal: #9a6700; --und: #cf222e; --conv: #0969da; --sem: #8250df;
      --asset: #bc4c00;
      --shadow: 0 1px 2px rgba(31,35,40,.06);
    }
  }
  :root[data-theme="light"] {
    --bg: #ffffff; --panel: #f6f8fa; --panel2: #eef1f5; --fg: #1f2328;
    --muted: #636c76; --line: #d0d7de; --accent: #0969da;
    --good: #1a7f37; --mal: #9a6700; --und: #cf222e; --conv: #0969da; --sem: #8250df;
    --asset: #bc4c00;
    --shadow: 0 1px 2px rgba(31,35,40,.06);
  }
  :root[data-theme="dark"] {
    --bg: #0f1117; --panel: #161922; --panel2: #1c202b; --fg: #e6e8ee;
    --muted: #8b90a0; --line: #262a36; --accent: #58a6ff;
    --good: #3fb950; --mal: #d29922; --und: #f85149; --conv: #58a6ff; --sem: #a371f7;
    --asset: #db6d28;
    --shadow: 0 1px 0 rgba(0,0,0,.3);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .layout { display: flex; align-items: flex-start; max-width: 1360px; margin: 0 auto; }
  .wrap { flex: 1; min-width: 0; max-width: 1120px; padding: 0 24px 72px; }
  header { display: flex; align-items: flex-start; justify-content: space-between;
           gap: 16px; padding: 24px 0 4px; }

  /* sticky left sidebar: section nav + at-a-glance stat, hidden on narrow screens */
  .sidebar { flex: none; width: 212px; align-self: flex-start; position: sticky; top: 0;
    max-height: 100vh; overflow-y: auto; padding: 26px 10px 24px 20px; }
  .side-title { color: var(--muted); font-size: 11px; text-transform: uppercase;
    letter-spacing: .08em; margin: 6px 8px 8px; }
  .nav-item { display: flex; align-items: baseline; justify-content: space-between; gap: 8px;
    padding: 7px 10px; border-radius: 8px; cursor: pointer; color: var(--muted);
    font-size: 13px; border-left: 2px solid transparent; }
  .nav-item:hover { background: var(--panel); color: var(--fg); }
  .nav-item.active { background: var(--panel); color: var(--fg); border-left-color: var(--accent); }
  .nav-item .v { font-variant-numeric: tabular-nums; font-size: 12px; font-weight: 600;
    color: var(--fg); }
  @media (max-width: 900px) { .sidebar { display: none; } }
  h1 { margin: 0; font-size: 20px; letter-spacing: -.01em; }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .07em;
       color: var(--muted); margin: 0; }
  .sub { color: var(--muted); font-size: 12.5px; margin: 8px 0 12px; max-width: 74ch; }
  .label { color: var(--muted); margin: 4px 0 0; font-size: 13px; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .92em; }
  .theme-btn { flex: none; background: var(--panel); color: var(--fg);
    border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px;
    font-size: 13px; cursor: pointer; box-shadow: var(--shadow); }
  .theme-btn:hover { border-color: var(--accent); }

  /* headline scorecards (clickable) */
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
           gap: 14px; margin: 16px 0 6px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
          padding: 14px 16px; box-shadow: var(--shadow); cursor: pointer; transition: border-color .12s; }
  .card:hover { border-color: var(--accent); }
  .score { font-size: 38px; font-weight: 700; line-height: 1.05;
           font-variant-numeric: tabular-nums; }
  .statlabel { color: var(--muted); font-size: 12px; text-transform: uppercase;
               letter-spacing: .06em; margin-top: 6px; }
  .statnote { color: var(--muted); font-size: 12px; margin-top: 2px; }

  details.panel { background: var(--panel); border: 1px solid var(--line);
    border-radius: 12px; padding: 12px 18px; margin-top: 14px; box-shadow: var(--shadow); }
  details.panel > summary { cursor: pointer; list-style: none; display: flex;
    align-items: baseline; gap: 10px; padding: 4px 0; }
  details.panel > summary::-webkit-details-marker { display: none; }
  details.panel > summary::before { content: "▸"; color: var(--muted);
    font-size: 11px; transition: transform .15s; }
  details.panel[open] > summary::before { transform: rotate(90deg); }
  .summary-stat { color: var(--muted); font-size: 12.5px; font-weight: 600; margin-left: auto; }

  .cats { display: grid; gap: 8px; margin-top: 6px; }
  .cat { display: grid; grid-template-columns: 96px 1fr 132px; align-items: center;
         gap: 12px; cursor: pointer; border-radius: 6px; padding: 1px 0; }
  .cat:hover .name { color: var(--fg); text-decoration: underline; }
  .cat .name { color: var(--muted); font-variant-numeric: tabular-nums; }
  .track { display: block; background: var(--panel2); border: 1px solid var(--line);
           border-radius: 6px; height: 15px; overflow: hidden; }
  .fill { display: block; height: 100%; border-radius: 5px 0 0 5px; transition: width .2s; }
  .cat .pct { text-align: right; font-variant-numeric: tabular-nums; }
  .cat .pct small { color: var(--muted); }

  .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 10px 0; }
  input, select { background: var(--bg); color: var(--fg); border: 1px solid var(--line);
    border-radius: 8px; padding: 8px 10px; font-size: 13px; }
  input[type=search] { flex: 1; min-width: 220px; }
  input:focus, select:focus { outline: none; border-color: var(--accent); }
  label.chk { color: var(--muted); font-size: 12.5px; display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }
  .count { color: var(--muted); font-size: 13px; }

  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line);
           font-variant-numeric: tabular-nums; }
  th { cursor: pointer; user-select: none; color: var(--muted); font-weight: 600;
       position: sticky; top: 0; background: var(--panel); font-size: 12px;
       text-transform: uppercase; letter-spacing: .04em; }
  th:hover { color: var(--fg); }
  tbody tr:hover { background: var(--panel2); }
  tr.clickable { cursor: pointer; }
  tr.group { cursor: pointer; }
  tr.group td { background: var(--panel2); font-family: ui-monospace, monospace;
    font-size: 12.5px; font-weight: 600; color: var(--fg); }
  tr.group:hover td { color: var(--accent); }
  tr.group .gtog { color: var(--muted); display: inline-block; width: 1.1em; }
  tr.group .gc { color: var(--muted); font-weight: 600; margin-left: 6px; }
  td.name, td.fam { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  td.file { color: var(--muted); font-family: ui-monospace, monospace; font-size: 12px; }
  td.reason { color: var(--fg); font-size: 12.5px; }
  .kind { color: var(--muted); }

  .minibar { display: inline-block; vertical-align: middle; width: 88px; height: 9px;
    background: var(--panel2); border: 1px solid var(--line); border-radius: 5px;
    overflow: hidden; margin-right: 8px; }
  .minibar > span { display: block; height: 100%; }

  .tag { display: inline-block; min-width: 1.7em; text-align: center; border-radius: 5px;
    font-weight: 700; font-size: 11px; padding: 2px 5px; }
  .tag.U { background: color-mix(in srgb, var(--und) 18%, transparent); color: var(--und); }
  .tag.M { background: color-mix(in srgb, var(--mal) 20%, transparent); color: var(--mal); }
  .tag.C { background: color-mix(in srgb, var(--conv) 18%, transparent); color: var(--conv); }
  .tag.S { background: color-mix(in srgb, var(--sem) 20%, transparent); color: var(--sem); }
  .tag.A { background: color-mix(in srgb, var(--asset) 22%, transparent); color: var(--asset); }
  .legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 4px 0 12px; font-size: 12.5px;
    color: var(--muted); }
  .legend span b { color: var(--fg); font-weight: 600; }
</style>
</head>
<body>
<div class="layout">
  <nav class="sidebar" id="sidebar">
    <div class="side-title">Sections</div>
    <a class="nav-item" data-target="panel-complete"><span>Completeness</span><span class="v" id="nv-complete"></span></a>
    <a class="nav-item" data-target="panel-assets"><span>Asset data</span><span class="v" id="nv-assets"></span></a>
    <a class="nav-item" data-target="panel-sem"><span>Semantic</span><span class="v" id="nv-sem"></span></a>
    <a class="nav-item" data-target="panel-uniform"><span>Uniformity</span><span class="v" id="nv-uniform"></span></a>
    <a class="nav-item" data-target="panel-family"><span>Families</span><span class="v" id="nv-family"></span></a>
    <a class="nav-item" data-target="doc-panel"><span>Doc comments</span><span class="v" id="nv-doc"></span></a>
    <a class="nav-item" data-target="worklist"><span>Worklist</span><span class="v" id="nv-work"></span></a>
  </nav>
<div class="wrap">
  <header>
    <div>
      <h1>SM64 Documentation — Current State</h1>
      <p class="label">__LABEL__</p>
    </div>
    <button class="theme-btn" id="theme" title="Toggle dark / light">🌓 Theme</button>
  </header>

  <div class="stats">
    <div class="card" id="card-complete">
      <div class="score" id="score" style="color:var(--good)"></div>
      <div class="statlabel">completeness</div>
      <div class="statnote">symbols with a real name →</div>
    </div>
    <div class="card" id="card-uniform">
      <div class="score" id="uscore" style="color:var(--conv)"></div>
      <div class="statlabel">uniformity</div>
      <div class="statnote">names following convention →</div>
    </div>
    <div class="card" id="card-sem">
      <div class="score" id="semscore" style="color:var(--sem)"></div>
      <div class="statlabel">wired up</div>
      <div class="statnote" id="semnote">entities linked to their impl →</div>
    </div>
    <div class="card" id="card-assets">
      <div class="score" id="ascore" style="color:var(--asset)"></div>
      <div class="statlabel">asset data named</div>
      <div class="statnote" id="asnote">geo / dl / anim / vtx →</div>
    </div>
  </div>

  <details class="panel" open id="panel-complete">
    <summary><h2>Completeness — do symbols have real names?</h2>
      <span class="summary-stat" id="complete-stat"></span></summary>
    <p class="sub">GOOD vs. MALFORMED (named but off-shape) vs. UNDOCUMENTED
      (a placeholder like <code>func_8024…</code>). This is <b>code</b> symbols;
      asset data is scored separately below. Click a kind to see its worklist.</p>
    <div class="cats" id="cats-complete"></div>
  </details>

  <details class="panel" open id="panel-assets">
    <summary><h2>Asset data — named vs. address placeholders</h2>
      <span class="summary-stat" id="assets-stat"></span></summary>
    <p class="sub">Display lists, geo layouts, animations, collision, vertices, textures
      and lighting under <code>actors/</code>, <code>levels/</code>, <code>data/</code> —
      ~20k symbols, the largest naming effort left. Most still carry a ROM-address
      placeholder like <code>birds_seg5_vertex_05000048</code>. Shown per asset type
      (lowest first); click one to see its worklist.</p>
    <div class="cats" id="cats-assets"></div>
  </details>

  <details class="panel" open id="panel-sem">
    <summary><h2>Semantic entities — are they wired up?</h2>
      <span class="summary-stat" id="sem-stat"></span></summary>
    <p class="sub">A prefix says what something is <em>called</em>; this asks whether
      it's <em>connected</em> — cross-referencing a symbol to its implementation
      (a handler, text table, level script, audio file). Click an entity to see its gaps.</p>
    <table>
      <thead><tr>
        <th>entity</th><th>cross-reference</th><th>linked</th><th>gaps</th>
      </tr></thead>
      <tbody id="sem-rows"></tbody>
    </table>
  </details>

  <details class="panel" open id="panel-uniform">
    <summary><h2>Uniformity — do those names follow convention?</h2>
      <span class="summary-stat" id="uniform-stat"></span></summary>
    <p class="sub">Of the real names, which follow the role-aware conventions
      (functions <code>snake_case</code>, types <code>PascalCase</code>, members
      <code>camelCase</code>, globals a <code>g</code>/<code>s</code> prefix,
      constants <code>UPPER_SNAKE</code>)? Click a kind to see its violations.</p>
    <div class="cats" id="cats-uniform"></div>
  </details>

  <details class="panel" id="panel-family">
    <summary><h2>Completeness by entity family</h2>
      <span class="summary-stat" id="family-stat"></span></summary>
    <p class="sub">Constants &amp; enums regrouped by prefix — ACT_* (Mario actions),
      SOUND_*, MODEL_*, … This is the <b>completeness</b> axis (what fraction have
      real, non-placeholder names), which genuinely varies per family — <em>not</em>
      the old per-family uniformity, which was tautological and was removed. Click a
      family to see its members.</p>
    <div class="controls">
      <label class="chk"><input type="checkbox" id="fam-all"> show fully-complete families</label>
      <span class="count" id="fam-count"></span>
    </div>
    <table>
      <thead><tr>
        <th data-fk="family">family</th>
        <th data-fk="count">count</th>
        <th data-fk="complete">completeness</th>
      </tr></thead>
      <tbody id="fam-rows"></tbody>
    </table>
  </details>

  <details class="panel" id="doc-panel">
    <summary><h2>Documentation comments</h2>
      <span class="summary-stat" id="doc-stat"></span></summary>
    <p class="sub">Functions with a <code>/** … */</code> doc comment above them —
      the decomp's house style. Informational (not every function needs prose), but
      the spread shows where prose docs are thin: audio is sparse, menu near-complete.</p>
    <div class="cats" id="cats-doc"></div>
  </details>

  <details class="panel" open id="worklist">
    <summary><h2>Worklist — what to fix</h2>
      <span class="summary-stat" id="worklist-stat"></span></summary>
    <div class="legend">
      <span><span class="tag U">U</span> <b>undocumented</b> placeholder name</span>
      <span><span class="tag M">M</span> <b>malformed</b> off-convention name</span>
      <span><span class="tag C">C</span> <b>convention</b> uniformity violation</span>
      <span><span class="tag S">S</span> <b>semantic</b> not wired to its impl</span>
      <span><span class="tag A">A</span> <b>asset</b> unnamed segment data</span>
    </div>
    <div class="controls">
      <input type="search" id="q" placeholder="filter by name or file…">
      <select id="axis">
        <option value="">all axes</option>
        <option value="completeness">completeness</option>
        <option value="convention">convention</option>
        <option value="semantic">semantic</option>
        <option value="asset">asset data</option>
      </select>
      <select id="kind"></select>
      <select id="famsel"></select>
      <label class="chk"><input type="checkbox" id="groupfile"> group by file</label>
      <button class="theme-btn" id="collapseall" style="display:none">collapse all</button>
      <button class="theme-btn" id="clear">clear</button>
      <span class="count" id="count"></span>
    </div>
    <table>
      <thead><tr>
        <th data-k="tag">!</th>
        <th data-k="kind">kind</th>
        <th data-k="family">group</th>
        <th data-k="name">name</th>
        <th data-k="file">file</th>
        <th data-k="line">line</th>
        <th data-k="reason">reason</th>
      </tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </details>
</div>
</div>
<script>
const DATA = __DATA__;
const KINDS = Object.keys(DATA.categories);

// --- theme toggle (CSS handles first paint via prefers-color-scheme) ---
const root = document.documentElement;
const saved = (function () { try { return localStorage.getItem('theme'); } catch (e) { return null; } })();
if (saved) root.dataset.theme = saved;
document.getElementById('theme').addEventListener('click', () => {
  const dark = getComputedStyle(root).getPropertyValue('--bg').trim().startsWith('#0');
  const next = (root.dataset.theme || (dark ? 'dark' : 'light')) === 'dark' ? 'light' : 'dark';
  root.dataset.theme = next;
  try { localStorage.setItem('theme', next); } catch (e) {}
});

function esc(s) { return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function barColor(r) { return r >= 0.9 ? 'var(--good)' : r >= 0.7 ? 'var(--mal)' : 'var(--und)'; }
function pctCell(ratio) {
  const p = (ratio * 100).toFixed(0);
  return `<span class="minibar"><span style="width:${(ratio*100).toFixed(1)}%;background:${barColor(ratio)}"></span></span>${p}%`;
}

// --- worklist focus: set filters and jump there (the heart of "click a thing") ---
const worklistEl = document.getElementById('worklist');
const q = document.getElementById('q'), kindSel = document.getElementById('kind');
const axisSel = document.getElementById('axis'), famSel = document.getElementById('famsel');
const tbody = document.getElementById('rows'), countEl = document.getElementById('count');

function setSel(sel, value) {
  // only apply if the option exists, else fall back to "all"
  sel.value = [...sel.options].some(o => o.value === value) ? value : '';
}
function focusWorklist({ axis = '', kind = '', group = '' } = {}) {
  q.value = '';
  setSel(axisSel, axis); setSel(kindSel, kind); setSel(famSel, group);
  render();
  worklistEl.open = true;
  worklistEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// headline scores (also clickable)
document.getElementById('score').textContent = (DATA.score * 100).toFixed(1) + '%';
document.getElementById('uscore').textContent = ((DATA.uniformity_score ?? 1) * 100).toFixed(1) + '%';
const SEM = (DATA.semantic_entities || []).slice().sort((a, b) => b.gaps - a.gaps);
const semTot = SEM.reduce((a, e) => a + e.members, 0);
const semLink = SEM.reduce((a, e) => a + e.linked, 0);
const semGaps = SEM.reduce((a, e) => a + e.gaps, 0);
document.getElementById('semscore').textContent =
  (semTot ? (100 * semLink / semTot) : 100).toFixed(1) + '%';
document.getElementById('semnote').textContent = `${semLink}/${semTot} linked →`;
document.getElementById('card-complete').addEventListener('click', () => focusWorklist({ axis: 'completeness' }));
document.getElementById('card-uniform').addEventListener('click', () => focusWorklist({ axis: 'convention' }));
document.getElementById('card-sem').addEventListener('click', () => focusWorklist({ axis: 'semantic' }));

// --- axis bars (click a kind -> worklist) ---
function bars(elId, axis, entries) {
  const el = document.getElementById(elId);
  for (const [name, good, total] of entries) {
    const ratio = total ? good / total : 1;
    const div = document.createElement('div');
    div.className = 'cat';
    div.innerHTML = `<span class="name">${name}</span>`
      + `<span class="track"><span class="fill" style="width:${(ratio*100).toFixed(1)}%;background:${barColor(ratio)}"></span></span>`
      + `<span class="pct">${(ratio*100).toFixed(1)}% <small>${good}/${total}</small></span>`;
    if (axis) div.addEventListener('click', () => focusWorklist({ axis, kind: name }));
    else div.style.cursor = 'default';
    el.appendChild(div);
  }
}
bars('cats-complete', 'completeness', KINDS.map(k => {
  const c = DATA.categories[k];
  return [k, c.GOOD, c.GOOD + c.MALFORMED + c.UNDOCUMENTED];
}));
bars('cats-uniform', 'convention', Object.keys(DATA.conventions || {}).map(k => {
  const c = DATA.conventions[k];
  return [k, c.CONFORMING, c.CONFORMING + c.VIOLATION];
}));
const compTotals = KINDS.reduce((a, k) => {
  const c = DATA.categories[k]; a.g += c.GOOD; a.t += c.GOOD + c.MALFORMED + c.UNDOCUMENTED; return a;
}, { g: 0, t: 0 });
document.getElementById('complete-stat').textContent =
  (compTotals.t - compTotals.g).toLocaleString() + ' to fix';
document.getElementById('uniform-stat').textContent =
  (DATA.violations || []).length.toLocaleString() + ' violations';

// --- semantic entities table (gaps first; click -> worklist) ---
document.getElementById('sem-stat').textContent = semGaps + ' gaps';
document.getElementById('sem-rows').innerHTML = SEM.map(e => {
  const ratio = e.members ? e.linked / e.members : 1;
  return `<tr class="clickable" data-entity="${esc(e.entity)}"><td class="fam">${esc(e.entity)}</td>`
    + `<td class="reason">${esc(e.link)}</td>`
    + `<td>${pctCell(ratio)} <small class="count">${e.linked}/${e.members}</small></td>`
    + `<td>${e.gaps}</td></tr>`;
}).join('') || '<tr><td colspan="4" class="count">none</td></tr>';
document.getElementById('sem-rows').addEventListener('click', ev => {
  const tr = ev.target.closest('tr[data-entity]');
  if (tr) focusWorklist({ axis: 'semantic', group: tr.dataset.entity });
});

// --- completeness-by-family table (sortable, hides 100% by default, click -> worklist) ---
const FAMILIES = Object.entries(DATA.families || {}).map(([family, c]) => ({
  family, count: c.count, complete: c.count ? c.good / c.count : 1,
}));
let famKey = 'complete', famDir = 1, famShowAll = false;
const famBody = document.getElementById('fam-rows');
const famCount = document.getElementById('fam-count');
const incomplete = FAMILIES.filter(f => f.complete < 1).length;
document.getElementById('family-stat').textContent = incomplete + ' need work';
function renderFamilies() {
  const view = (famShowAll ? FAMILIES : FAMILIES.filter(f => f.complete < 1)).slice();
  view.sort((a, b) => {
    let x = a[famKey], y = b[famKey];
    return (x < y ? -1 : x > y ? 1 : 0) * famDir;
  });
  famCount.textContent = `${view.length} of ${FAMILIES.length} families`;
  famBody.innerHTML = view.map(f =>
    `<tr class="clickable" data-group="${esc(f.family)}"><td class="fam">${esc(f.family)}</td>`
    + `<td>${f.count}</td><td>${pctCell(f.complete)}</td></tr>`
  ).join('') || '<tr><td colspan="3" class="count">all families fully complete 🎉</td></tr>';
}
document.querySelectorAll('th[data-fk]').forEach(th => th.addEventListener('click', () => {
  const k = th.dataset.fk;
  if (k === famKey) famDir = -famDir; else { famKey = k; famDir = 1; }
  renderFamilies();
}));
document.getElementById('fam-all').addEventListener('change', e => { famShowAll = e.target.checked; renderFamilies(); });
famBody.addEventListener('click', ev => {
  const tr = ev.target.closest('tr[data-group]');
  if (tr) focusWorklist({ group: tr.dataset.group });
});
renderFamilies();

// --- documentation-comment coverage (informational; hidden if not measured) ---
const DOC = DATA.doc_comments;
if (DOC && DOC.total.functions) {
  document.getElementById('doc-stat').textContent =
    Math.round(100 * DOC.total.documented / DOC.total.functions) + '% have a /** */';
  bars('cats-doc', '', Object.entries(DOC.by_area)
    .sort((a, b) => (a[1].documented / a[1].functions) - (b[1].documented / b[1].functions))
    .map(([area, c]) => [area, c.documented, c.functions]));
} else {
  document.getElementById('doc-panel').style.display = 'none';
}

// --- asset data (named vs. ROM-address placeholders), by asset type ---
const AS = DATA.assets;
if (AS && AS.total.total) {
  const ar = AS.total.named / AS.total.total;
  document.getElementById('ascore').textContent = (ar * 100).toFixed(0) + '%';
  document.getElementById('asnote').textContent =
    AS.total.named.toLocaleString() + '/' + AS.total.total.toLocaleString() + ' named →';
  document.getElementById('assets-stat').textContent =
    (AS.total.total - AS.total.named).toLocaleString() + ' to name';
  bars('cats-assets', 'asset',
    Object.entries(AS.by_type).map(([t, c]) => [t, c.named, c.total]));
  document.getElementById('card-assets').addEventListener('click', () => focusWorklist({ axis: 'asset' }));
} else {
  document.getElementById('panel-assets').style.display = 'none';
  document.getElementById('card-assets').style.display = 'none';
}

// --- worklist (all axes combined) ---
const rows = [
  ...DATA.needs_attention.map(r => ({
    tag: r.classification === 'UNDOCUMENTED' ? 'U' : 'M', axis: 'completeness',
    kind: r.kind, family: r.family || '', name: r.name, file: r.file, line: r.line,
    reason: r.reason || (r.classification === 'UNDOCUMENTED' ? 'undocumented' : 'malformed'),
  })),
  ...(DATA.violations || []).map(r => ({
    tag: 'C', axis: 'convention',
    kind: r.kind, family: r.family || '', name: r.name, file: r.file, line: r.line,
    reason: r.reason,
  })),
  ...(DATA.semantic_findings || []).map(r => ({
    tag: 'S', axis: 'semantic',
    kind: '', family: r.entity, name: r.name, file: r.file, line: r.line,
    reason: r.detail,
  })),
  ...(DATA.asset_findings || []).map(r => ({
    tag: 'A', axis: 'asset',
    kind: r.kind, family: '', name: r.name, file: r.file, line: r.line,
    reason: r.reason || 'ROM-address placeholder — give it a real name',
  })),
];

let sortKey = 'file', sortDir = 1;

const allKinds = [...new Set(rows.map(r => r.kind))].filter(Boolean).sort();
kindSel.innerHTML = '<option value="">all kinds</option>'
  + allKinds.map(k => `<option value="${k}">${k}</option>`).join('');
// group filter = every distinct group present in the worklist (prefix families + entities)
const groupOpts = [...new Set(rows.map(r => r.family).filter(Boolean))].sort();
famSel.innerHTML = '<option value="">all groups</option>'
  + groupOpts.map(f => `<option value="${esc(f)}">${esc(f)}</option>`).join('');

const groupChk = document.getElementById('groupfile');
const collapseBtn = document.getElementById('collapseall');
const collapsed = new Set();   // files collapsed in group-by-file mode
let groupedFiles = [];         // files in the current grouped view (for collapse-all)

function rowHtml(r) {
  return `<tr><td><span class="tag ${r.tag}">${r.tag}</span></td>`
    + `<td class="kind">${r.kind}</td>`
    + `<td class="fam">${esc(r.family)}</td>`
    + `<td class="name">${esc(r.name)}</td>`
    + `<td class="file">${esc(r.file)}</td>`
    + `<td>${r.line}</td>`
    + `<td class="reason">${esc(r.reason)}</td></tr>`;
}

function render() {
  const term = q.value.toLowerCase(), fk = kindSel.value, fa = axisSel.value, ff = famSel.value;
  let view = rows.filter(r =>
    (!fk || r.kind === fk) && (!fa || r.axis === fa) && (!ff || r.family === ff) &&
    (!term || r.name.toLowerCase().includes(term) || r.file.toLowerCase().includes(term)));
  document.getElementById('worklist-stat').textContent = rows.length.toLocaleString() + ' items';
  const MAX = 2000;

  if (groupChk.checked) {
    // group rows under collapsible file headers, files with the most to fix first
    const groups = new Map();
    for (const r of view) {
      if (!groups.has(r.file)) groups.set(r.file, []);
      groups.get(r.file).push(r);
    }
    const files = [...groups.entries()].sort((a, b) =>
      b[1].length - a[1].length || (a[0] < b[0] ? -1 : 1));
    groupedFiles = files.map(f => f[0]);
    countEl.textContent = files.length.toLocaleString() + ' files · '
      + view.length.toLocaleString() + ' of ' + rows.length.toLocaleString() + ' items';
    collapseBtn.textContent = collapsed.size ? 'expand all' : 'collapse all';
    let out = '', shown = 0, capped = false;
    for (const [file, frows] of files) {
      const open = !collapsed.has(file);
      out += `<tr class="group" data-file="${esc(file)}"><td colspan="7">`
        + `<span class="gtog">${open ? '▾' : '▸'}</span>${esc(file)}`
        + `<span class="gc">${frows.length}</span></td></tr>`;
      if (open) {
        frows.sort((a, b) => a.line - b.line);
        for (const r of frows) {
          if (shown >= MAX) { capped = true; break; }
          out += rowHtml(r); shown++;
        }
      }
      if (capped) break;
    }
    tbody.innerHTML = (out + (capped
      ? `<tr><td colspan="7" class="count">… more rows hidden (collapse files or narrow the filter)</td></tr>`
      : '')) || '<tr><td colspan="7" class="count">nothing matches</td></tr>';
  } else {
    view.sort((a, b) => {
      let x = a[sortKey], y = b[sortKey];
      return (x < y ? -1 : x > y ? 1 : 0) * sortDir;
    });
    countEl.textContent = view.length.toLocaleString() + ' of '
      + rows.length.toLocaleString() + ' items';
    tbody.innerHTML = view.slice(0, MAX).map(rowHtml).join('') + (view.length > MAX
      ? `<tr><td colspan="7" class="count">… ${view.length - MAX} more (narrow the filter)</td></tr>` : '');
  }
}

document.querySelectorAll('th[data-k]').forEach(th => th.addEventListener('click', () => {
  const k = th.dataset.k;
  if (k === sortKey) sortDir = -sortDir; else { sortKey = k; sortDir = 1; }
  render();
}));
[q, kindSel, axisSel, famSel].forEach(el => el.addEventListener('input', render));
groupChk.addEventListener('change', () => {
  collapseBtn.style.display = groupChk.checked ? '' : 'none';
  render();
});
collapseBtn.addEventListener('click', () => {
  if (collapsed.size) collapsed.clear();              // some collapsed -> expand all
  else groupedFiles.forEach(f => collapsed.add(f));   // all open -> collapse all
  render();
});
tbody.addEventListener('click', ev => {
  const g = ev.target.closest('tr.group');
  if (!g) return;
  const f = g.dataset.file;
  if (collapsed.has(f)) collapsed.delete(f); else collapsed.add(f);
  render();
});
document.getElementById('clear').addEventListener('click', () => {
  q.value = ''; axisSel.value = ''; kindSel.value = ''; famSel.value = '';
  groupChk.checked = false; collapseBtn.style.display = 'none'; collapsed.clear();
  render();
});
render();

// --- sidebar: at-a-glance stats, click-to-jump, scroll-spy ---
function navStat(id, val) { const e = document.getElementById(id); if (e) e.textContent = val; }
navStat('nv-complete', (DATA.score * 100).toFixed(0) + '%');
navStat('nv-assets', (AS && AS.total.total) ? (100 * AS.total.named / AS.total.total).toFixed(0) + '%' : '—');
navStat('nv-sem', (semTot ? 100 * semLink / semTot : 100).toFixed(0) + '%');
navStat('nv-uniform', ((DATA.uniformity_score ?? 1) * 100).toFixed(0) + '%');
navStat('nv-family', String(incomplete));
navStat('nv-doc', (DOC && DOC.total.functions) ? Math.round(100 * DOC.total.documented / DOC.total.functions) + '%' : '—');
navStat('nv-work', rows.length.toLocaleString());
const navItems = [...document.querySelectorAll('.nav-item')];
navItems.forEach(a => a.addEventListener('click', () => {
  const el = document.getElementById(a.dataset.target);
  if (!el) return;
  if (el.tagName === 'DETAILS') el.open = true;
  el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}));
if (typeof IntersectionObserver !== 'undefined') {
  const spy = new IntersectionObserver(es => es.forEach(en => {
    if (en.isIntersecting) navItems.forEach(a => a.classList.toggle('active', a.dataset.target === en.target.id));
  }), { rootMargin: '-12% 0px -80% 0px' });
  navItems.forEach(a => { const el = document.getElementById(a.dataset.target); if (el) spy.observe(el); });
}
</script>
</body>
</html>
"""


def render(data: dict, label: str = "") -> str:
    # Escape "</" so an embedded string can't close the <script> early.
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    safe_label = html.escape(label or "n64decomp/sm64")
    return _TEMPLATE.replace("__LABEL__", safe_label).replace("__DATA__", blob)
