"""Render a self-contained current-state HTML report.

One static file: the result JSON is embedded and a little vanilla JS renders it,
so it drops straight onto GitHub Pages with no external assets. The page has a
dark/light toggle, three headline scores (completeness, uniformity, wired-up),
per-kind bars for the two name-quality axes, a completeness-by-family table, a
semantic cross-reference table, and one sortable/filterable worklist.

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
    --shadow: 0 1px 0 rgba(0,0,0,.3);
  }
  @media (prefers-color-scheme: light) {
    :root:not([data-theme]) {
      --bg: #ffffff; --panel: #f6f8fa; --panel2: #eef1f5; --fg: #1f2328;
      --muted: #636c76; --line: #d0d7de; --accent: #0969da;
      --good: #1a7f37; --mal: #9a6700; --und: #cf222e; --conv: #0969da; --sem: #8250df;
      --shadow: 0 1px 2px rgba(31,35,40,.06);
    }
  }
  :root[data-theme="light"] {
    --bg: #ffffff; --panel: #f6f8fa; --panel2: #eef1f5; --fg: #1f2328;
    --muted: #636c76; --line: #d0d7de; --accent: #0969da;
    --good: #1a7f37; --mal: #9a6700; --und: #cf222e; --conv: #0969da; --sem: #8250df;
    --shadow: 0 1px 2px rgba(31,35,40,.06);
  }
  :root[data-theme="dark"] {
    --bg: #0f1117; --panel: #161922; --panel2: #1c202b; --fg: #e6e8ee;
    --muted: #8b90a0; --line: #262a36; --accent: #58a6ff;
    --good: #3fb950; --mal: #d29922; --und: #f85149; --conv: #58a6ff; --sem: #a371f7;
    --shadow: 0 1px 0 rgba(0,0,0,.3);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1120px; margin: 0 auto; padding: 0 24px 72px; }
  header { display: flex; align-items: flex-start; justify-content: space-between;
           gap: 16px; padding: 28px 0 6px; }
  h1 { margin: 0; font-size: 20px; letter-spacing: -.01em; }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .07em;
       color: var(--muted); margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 12.5px; margin: 0 0 14px; max-width: 70ch; }
  .label { color: var(--muted); margin: 4px 0 0; font-size: 13px; }
  .theme-btn { flex: none; background: var(--panel); color: var(--fg);
    border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px;
    font-size: 13px; cursor: pointer; box-shadow: var(--shadow); }
  .theme-btn:hover { border-color: var(--accent); }

  /* headline scorecards */
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
           margin: 18px 0 8px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
          padding: 16px 18px; box-shadow: var(--shadow); }
  .score { font-size: 42px; font-weight: 700; line-height: 1.05;
           font-variant-numeric: tabular-nums; }
  .statlabel { color: var(--muted); font-size: 12px; text-transform: uppercase;
               letter-spacing: .06em; margin-top: 8px; }
  .statnote { color: var(--muted); font-size: 12px; margin-top: 2px; }

  section.panel { background: var(--panel); border: 1px solid var(--line);
    border-radius: 12px; padding: 18px 20px; margin-top: 18px; box-shadow: var(--shadow); }

  .cats { display: grid; gap: 9px; }
  .cat { display: grid; grid-template-columns: 96px 1fr 132px; align-items: center; gap: 12px; }
  .cat .name { color: var(--muted); font-variant-numeric: tabular-nums; }
  .track { display: block; background: var(--panel2); border: 1px solid var(--line);
           border-radius: 6px; height: 16px; overflow: hidden; }
  .fill { display: block; height: 100%; border-radius: 5px 0 0 5px; transition: width .2s; }
  .cat .pct { text-align: right; font-variant-numeric: tabular-nums; }
  .cat .pct small { color: var(--muted); }

  .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; }
  input, select { background: var(--bg); color: var(--fg); border: 1px solid var(--line);
    border-radius: 8px; padding: 8px 10px; font-size: 13px; }
  input[type=search] { flex: 1; min-width: 220px; }
  input:focus, select:focus { outline: none; border-color: var(--accent); }
  .count { color: var(--muted); font-size: 13px; }

  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line);
           font-variant-numeric: tabular-nums; }
  th { cursor: pointer; user-select: none; color: var(--muted); font-weight: 600;
       position: sticky; top: 0; background: var(--panel); font-size: 12px;
       text-transform: uppercase; letter-spacing: .04em; }
  th:hover { color: var(--fg); }
  tbody tr:hover { background: var(--panel2); }
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
  .legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 0 0 14px; font-size: 12.5px;
    color: var(--muted); }
  .legend span b { color: var(--fg); font-weight: 600; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>SM64 Documentation — Current State</h1>
      <p class="label">__LABEL__</p>
    </div>
    <button class="theme-btn" id="theme" title="Toggle dark / light">🌓 Theme</button>
  </header>

  <div class="stats">
    <div class="card">
      <div class="score" id="score" style="color:var(--good)"></div>
      <div class="statlabel">completeness</div>
      <div class="statnote">symbols with a real name</div>
    </div>
    <div class="card">
      <div class="score" id="uscore" style="color:var(--conv)"></div>
      <div class="statlabel">uniformity</div>
      <div class="statnote">those names following convention</div>
    </div>
    <div class="card">
      <div class="score" id="semscore" style="color:var(--sem)"></div>
      <div class="statlabel">wired up</div>
      <div class="statnote" id="semnote">entities linked to their implementation</div>
    </div>
  </div>

  <section class="panel">
    <h2>Completeness — do symbols have real names?</h2>
    <p class="sub">Each named symbol the decomp exposes is GOOD, MALFORMED (named but
      off-shape), or UNDOCUMENTED (a placeholder like <code>func_8024…</code>). Bars
      show the GOOD fraction per kind.</p>
    <div class="cats" id="cats-complete"></div>
  </section>

  <section class="panel">
    <h2>Completeness by entity family</h2>
    <p class="sub">Constants &amp; enums regrouped by prefix — ACT_* (Mario actions),
      SOUND_*, MODEL_*, … — so a domain entity is navigable. This is the
      <b>completeness</b> axis (what fraction have real, non-placeholder names), which
      genuinely varies per family. It is <em>not</em> the old per-family uniformity,
      which was tautological (a family is its prefix) and was removed.</p>
    <div class="controls"><span class="count" id="fam-count"></span></div>
    <table>
      <thead><tr>
        <th data-fk="family">family</th>
        <th data-fk="count">count</th>
        <th data-fk="complete">completeness</th>
      </tr></thead>
      <tbody id="fam-rows"></tbody>
    </table>
  </section>

  <section class="panel">
    <h2>Uniformity — do those names follow convention?</h2>
    <p class="sub">Of the names that are real, do they follow the project's role-aware
      conventions (functions <code>snake_case</code>, types <code>PascalCase</code>,
      members <code>camelCase</code>, globals a <code>g</code>/<code>s</code> prefix,
      constants <code>UPPER_SNAKE</code>)?</p>
    <div class="cats" id="cats-uniform"></div>
  </section>

  <section class="panel">
    <h2>Semantic entities — are they wired up?</h2>
    <p class="sub">A prefix says what something is <em>called</em>; a semantic entity
      asks whether it's <em>connected</em> — by cross-referencing a symbol to its
      implementation (a handler, a text table, a level script, an audio file). Not
      tautological: the check crosses kinds and files.</p>
    <table>
      <thead><tr>
        <th>entity</th><th>cross-reference</th><th>linked</th><th>gaps</th>
      </tr></thead>
      <tbody id="sem-rows"></tbody>
    </table>
  </section>

  <section class="panel">
    <h2>Worklist — what to fix</h2>
    <div class="legend">
      <span><span class="tag U">U</span> <b>undocumented</b> placeholder name</span>
      <span><span class="tag M">M</span> <b>malformed</b> off-convention name</span>
      <span><span class="tag C">C</span> <b>convention</b> uniformity violation</span>
      <span><span class="tag S">S</span> <b>semantic</b> not wired to its implementation</span>
    </div>
    <div class="controls">
      <input type="search" id="q" placeholder="filter by name or file…">
      <select id="axis">
        <option value="">all axes</option>
        <option value="completeness">completeness</option>
        <option value="convention">convention</option>
        <option value="semantic">semantic</option>
      </select>
      <select id="kind"></select>
      <select id="famsel"></select>
      <span class="count" id="count"></span>
    </div>
    <table>
      <thead><tr>
        <th data-k="tag">!</th>
        <th data-k="kind">kind</th>
        <th data-k="family">family</th>
        <th data-k="name">name</th>
        <th data-k="file">file</th>
        <th data-k="line">line</th>
        <th data-k="reason">reason</th>
      </tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </section>
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

// headline scores
document.getElementById('score').textContent = (DATA.score * 100).toFixed(1) + '%';
document.getElementById('uscore').textContent = ((DATA.uniformity_score ?? 1) * 100).toFixed(1) + '%';
const SEM = DATA.semantic_entities || [];
const semTot = SEM.reduce((a, e) => a + e.members, 0);
const semLink = SEM.reduce((a, e) => a + e.linked, 0);
document.getElementById('semscore').textContent =
  (semTot ? (100 * semLink / semTot) : 100).toFixed(1) + '%';
document.getElementById('semnote').textContent =
  `${semLink}/${semTot} across ${SEM.length} entities`;

// --- axis bars ---
function bars(elId, entries) {
  const el = document.getElementById(elId);
  for (const [name, good, total] of entries) {
    const ratio = total ? good / total : 1;
    const div = document.createElement('div');
    div.className = 'cat';
    div.innerHTML = `<span class="name">${name}</span>`
      + `<span class="track"><span class="fill" style="width:${(ratio*100).toFixed(1)}%;background:${barColor(ratio)}"></span></span>`
      + `<span class="pct">${(ratio*100).toFixed(1)}% <small>${good}/${total}</small></span>`;
    el.appendChild(div);
  }
}
bars('cats-complete', KINDS.map(k => {
  const c = DATA.categories[k];
  return [k, c.GOOD, c.GOOD + c.MALFORMED + c.UNDOCUMENTED];
}));
bars('cats-uniform', Object.keys(DATA.conventions || {}).map(k => {
  const c = DATA.conventions[k];
  return [k, c.CONFORMING, c.CONFORMING + c.VIOLATION];
}));

// --- semantic entities table ---
document.getElementById('sem-rows').innerHTML = SEM.map(e => {
  const ratio = e.members ? e.linked / e.members : 1;
  return `<tr><td class="fam">${esc(e.entity)}</td>`
    + `<td class="reason">${esc(e.link)}</td>`
    + `<td>${pctCell(ratio)} <small class="count">${e.linked}/${e.members}</small></td>`
    + `<td>${e.gaps}</td></tr>`;
}).join('') || '<tr><td colspan="4" class="count">none</td></tr>';

// --- completeness-by-family table (sortable) ---
const FAMILIES = Object.entries(DATA.families || {}).map(([family, c]) => ({
  family, count: c.count, complete: c.count ? c.good / c.count : 1,
}));
let famKey = 'complete', famDir = 1;
const famBody = document.getElementById('fam-rows');
document.getElementById('fam-count').textContent = FAMILIES.length + ' families';
function renderFamilies() {
  FAMILIES.sort((a, b) => {
    let x = a[famKey], y = b[famKey];
    return (x < y ? -1 : x > y ? 1 : 0) * famDir;
  });
  famBody.innerHTML = FAMILIES.map(f =>
    `<tr><td class="fam">${esc(f.family)}</td><td>${f.count}</td>`
    + `<td>${pctCell(f.complete)}</td></tr>`
  ).join('');
}
document.querySelectorAll('th[data-fk]').forEach(th => th.addEventListener('click', () => {
  const k = th.dataset.fk;
  if (k === famKey) famDir = -famDir; else { famKey = k; famDir = 1; }
  renderFamilies();
}));
renderFamilies();

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
];

let sortKey = 'file', sortDir = 1;
const q = document.getElementById('q'), kindSel = document.getElementById('kind');
const axisSel = document.getElementById('axis'), famSel = document.getElementById('famsel');
const tbody = document.getElementById('rows'), countEl = document.getElementById('count');

const allKinds = [...new Set(rows.map(r => r.kind))].filter(Boolean).sort();
kindSel.innerHTML = '<option value="">all kinds</option>'
  + allKinds.map(k => `<option value="${k}">${k}</option>`).join('');
const famOpts = FAMILIES.slice().sort((a, b) => b.count - a.count).map(f => f.family);
famSel.innerHTML = '<option value="">all families</option>'
  + famOpts.map(f => `<option value="${esc(f)}">${esc(f)}</option>`).join('');

function render() {
  const term = q.value.toLowerCase(), fk = kindSel.value, fa = axisSel.value, ff = famSel.value;
  let view = rows.filter(r =>
    (!fk || r.kind === fk) && (!fa || r.axis === fa) && (!ff || r.family === ff) &&
    (!term || r.name.toLowerCase().includes(term) || r.file.toLowerCase().includes(term)));
  view.sort((a, b) => {
    let x = a[sortKey], y = b[sortKey];
    return (x < y ? -1 : x > y ? 1 : 0) * sortDir;
  });
  countEl.textContent = view.length.toLocaleString() + ' of '
    + rows.length.toLocaleString() + ' items';
  const MAX = 2000;
  tbody.innerHTML = view.slice(0, MAX).map(r =>
    `<tr><td><span class="tag ${r.tag}">${r.tag}</span></td>`
    + `<td class="kind">${r.kind}</td>`
    + `<td class="fam">${esc(r.family)}</td>`
    + `<td class="name">${esc(r.name)}</td>`
    + `<td class="file">${esc(r.file)}</td>`
    + `<td>${r.line}</td>`
    + `<td class="reason">${esc(r.reason)}</td></tr>`
  ).join('') + (view.length > MAX
    ? `<tr><td colspan="7" class="count">… ${view.length - MAX} more (narrow the filter)</td></tr>` : '');
}

document.querySelectorAll('th[data-k]').forEach(th => th.addEventListener('click', () => {
  const k = th.dataset.k;
  if (k === sortKey) sortDir = -sortDir; else { sortKey = k; sortDir = 1; }
  render();
}));
[q, kindSel, axisSel, famSel].forEach(el => el.addEventListener('input', render));
render();
</script>
</body>
</html>
"""


def render(data: dict, label: str = "") -> str:
    # Escape "</" so an embedded string can't close the <script> early.
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    safe_label = html.escape(label or "n64decomp/sm64")
    return _TEMPLATE.replace("__LABEL__", safe_label).replace("__DATA__", blob)
