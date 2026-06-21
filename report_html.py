"""Render a self-contained current-state HTML report.

One static file: the result JSON is embedded, and a little vanilla JS renders the
two axes -- completeness and uniformity -- as per-category bars, an entity-family
table (constants/enums grouped by prefix, so ACT_*/SOUND_*/... are named things),
and a single sortable/filterable worklist. No external assets, so it drops
straight onto GitHub Pages.
"""

import json

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SM64 Documentation — Current State</title>
<style>
  :root {{
    --bg: #0f1117; --panel: #181b24; --fg: #e6e8ee; --muted: #8b90a0;
    --good: #3fb950; --mal: #d29922; --und: #f85149; --conv: #58a6ff; --line: #262a36;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 0 24px 60px; }}
  header {{ padding: 28px 0 4px; }}
  h1 {{ margin: 0; font-size: 20px; }}
  h2 {{ font-size: 13px; text-transform: uppercase; letter-spacing: .06em;
        color: var(--muted); margin: 26px 0 10px; }}
  .label {{ color: var(--muted); margin: 4px 0 0; font-size: 13px; }}
  .stats {{ display: flex; gap: 40px; margin: 16px 0 4px; }}
  .score {{ font-size: 52px; font-weight: 700; line-height: 1;
            font-variant-numeric: tabular-nums; }}
  .statlabel {{ color: var(--muted); font-size: 12px; text-transform: uppercase;
                letter-spacing: .06em; margin-top: 6px; }}
  .cats {{ display: grid; gap: 8px; }}
  .cat {{ display: grid; grid-template-columns: 90px 1fr 130px;
          align-items: center; gap: 12px; }}
  .cat .name {{ color: var(--muted); }}
  .track {{ display: block; background: var(--line); border-radius: 6px;
            height: 14px; overflow: hidden; }}
  .fill {{ display: block; height: 100%; background: var(--good); }}
  .cat .pct {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .cat .pct small {{ color: var(--muted); }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
               margin-bottom: 12px; }}
  input, select {{ background: var(--panel); color: var(--fg);
    border: 1px solid var(--line); border-radius: 6px; padding: 7px 10px; font-size: 13px; }}
  input[type=search] {{ flex: 1; min-width: 200px; }}
  .count {{ color: var(--muted); font-size: 13px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line);
            font-variant-numeric: tabular-nums; }}
  th {{ cursor: pointer; user-select: none; color: var(--muted); font-weight: 600;
        position: sticky; top: 0; background: var(--bg); }}
  th:hover {{ color: var(--fg); }}
  td.name, td.fam {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  td.file {{ color: var(--muted); font-family: ui-monospace, monospace; font-size: 12px; }}
  td.reason {{ color: var(--muted); font-size: 12px; }}
  .minibar {{ display: inline-block; vertical-align: middle; width: 90px; height: 8px;
              background: var(--line); border-radius: 4px; overflow: hidden; margin-right: 8px; }}
  .minibar > span {{ display: block; height: 100%; background: var(--good); }}
  .tag {{ display: inline-block; width: 1.6em; text-align: center; border-radius: 4px;
          font-weight: 700; font-size: 11px; padding: 1px 0; }}
  .tag.U {{ background: rgba(248,81,73,.18); color: var(--und); }}
  .tag.M {{ background: rgba(210,153,34,.18); color: var(--mal); }}
  .tag.C {{ background: rgba(88,166,255,.18); color: var(--conv); }}
  .kind {{ color: var(--muted); }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>SM64 Documentation — Current State</h1>
    <p class="label">{label}</p>
    <div class="stats">
      <div><div class="score" id="score"></div><div class="statlabel">completeness</div></div>
      <div><div class="score" id="uscore"></div><div class="statlabel">uniformity</div></div>
    </div>
  </header>

  <h2>Completeness — do symbols have real names?</h2>
  <div class="cats" id="cats-complete"></div>

  <h2>Uniformity — do those names follow convention?</h2>
  <div class="cats" id="cats-uniform"></div>

  <h2>Entities — constant &amp; enum families (grouped by prefix)</h2>
  <div class="controls"><span class="count" id="fam-count"></span></div>
  <table>
    <thead><tr>
      <th data-fk="family">family</th>
      <th data-fk="count">count</th>
      <th data-fk="complete">completeness</th>
      <th data-fk="uniform">uniformity</th>
    </tr></thead>
    <tbody id="fam-rows"></tbody>
  </table>

  <h2>Worklist</h2>
  <div class="controls">
    <input type="search" id="q" placeholder="filter by name or file…">
    <select id="axis">
      <option value="">both axes</option>
      <option value="completeness">completeness</option>
      <option value="convention">convention</option>
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
</div>
<script>
const DATA = {data};
const KINDS = Object.keys(DATA.categories);

document.getElementById('score').textContent = (DATA.score * 100).toFixed(1) + '%';
document.getElementById('uscore').textContent =
  ((DATA.uniformity_score ?? 1) * 100).toFixed(1) + '%';

function esc(s) {{ return String(s).replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c])); }}
function pctCell(ratio) {{
  const p = (ratio * 100).toFixed(0);
  return `<span class="minibar"><span style="width:${{(ratio*100).toFixed(1)}}%"></span></span>${{p}}%`;
}}

// --- axis bars ---
function bars(elId, entries) {{
  const el = document.getElementById(elId);
  for (const [name, good, total] of entries) {{
    const ratio = total ? good / total : 1;
    const div = document.createElement('div');
    div.className = 'cat';
    div.innerHTML = `<span class="name">${{name}}</span>`
      + `<span class="track"><span class="fill" style="width:${{(ratio*100).toFixed(1)}}%"></span></span>`
      + `<span class="pct">${{(ratio*100).toFixed(1)}}% <small>${{good}}/${{total}}</small></span>`;
    el.appendChild(div);
  }}
}}
bars('cats-complete', KINDS.map(k => {{
  const c = DATA.categories[k];
  return [k, c.GOOD, c.GOOD + c.MALFORMED + c.UNDOCUMENTED];
}}));
bars('cats-uniform', Object.keys(DATA.conventions || {{}}).map(k => {{
  const c = DATA.conventions[k];
  return [k, c.CONFORMING, c.CONFORMING + c.VIOLATION];
}}));

// --- entity families table (sortable) ---
const FAMILIES = Object.entries(DATA.families || {{}}).map(([family, c]) => ({{
  family, count: c.count,
  complete: c.count ? c.good / c.count : 1,
  uniform: c.good ? c.conforming / c.good : 1,
}}));
let famKey = 'complete', famDir = 1;
const famBody = document.getElementById('fam-rows');
document.getElementById('fam-count').textContent = FAMILIES.length + ' families';
function renderFamilies() {{
  FAMILIES.sort((a, b) => {{
    let x = a[famKey], y = b[famKey];
    return (x < y ? -1 : x > y ? 1 : 0) * famDir;
  }});
  famBody.innerHTML = FAMILIES.map(f =>
    `<tr><td class="fam">${{esc(f.family)}}</td><td>${{f.count}}</td>`
    + `<td>${{pctCell(f.complete)}}</td><td>${{pctCell(f.uniform)}}</td></tr>`
  ).join('');
}}
document.querySelectorAll('th[data-fk]').forEach(th => th.addEventListener('click', () => {{
  const k = th.dataset.fk;
  if (k === famKey) famDir = -famDir; else {{ famKey = k; famDir = (k === 'family') ? 1 : 1; }}
  renderFamilies();
}}));
renderFamilies();

// --- worklist (both axes combined) ---
const tagReason = {{UNDOCUMENTED: 'undocumented', MALFORMED: 'malformed'}};
const rows = [
  ...DATA.needs_attention.map(r => ({{
    tag: r.classification === 'UNDOCUMENTED' ? 'U' : 'M', axis: 'completeness',
    kind: r.kind, family: r.family || '', name: r.name, file: r.file, line: r.line,
    reason: tagReason[r.classification] || '',
  }})),
  ...(DATA.violations || []).map(r => ({{
    tag: 'C', axis: 'convention',
    kind: r.kind, family: r.family || '', name: r.name, file: r.file, line: r.line,
    reason: r.reason,
  }})),
];

let sortKey = 'file', sortDir = 1;
const q = document.getElementById('q'), kindSel = document.getElementById('kind');
const axisSel = document.getElementById('axis'), famSel = document.getElementById('famsel');
const tbody = document.getElementById('rows'), countEl = document.getElementById('count');

const allKinds = [...new Set(rows.map(r => r.kind))].sort();
kindSel.innerHTML = '<option value="">all kinds</option>'
  + allKinds.map(k => `<option value="${{k}}">${{k}}</option>`).join('');
// family options, by descending member count (most prominent first)
const famOpts = FAMILIES.slice().sort((a, b) => b.count - a.count).map(f => f.family);
famSel.innerHTML = '<option value="">all families</option>'
  + famOpts.map(f => `<option value="${{esc(f)}}">${{esc(f)}}</option>`).join('');

function render() {{
  const term = q.value.toLowerCase(), fk = kindSel.value, fa = axisSel.value, ff = famSel.value;
  let view = rows.filter(r =>
    (!fk || r.kind === fk) && (!fa || r.axis === fa) && (!ff || r.family === ff) &&
    (!term || r.name.toLowerCase().includes(term) || r.file.toLowerCase().includes(term)));
  view.sort((a, b) => {{
    let x = a[sortKey], y = b[sortKey];
    return (x < y ? -1 : x > y ? 1 : 0) * sortDir;
  }});
  countEl.textContent = view.length.toLocaleString() + ' of '
    + rows.length.toLocaleString() + ' items';
  const MAX = 2000;
  tbody.innerHTML = view.slice(0, MAX).map(r =>
    `<tr><td><span class="tag ${{r.tag}}">${{r.tag}}</span></td>`
    + `<td class="kind">${{r.kind}}</td>`
    + `<td class="fam">${{esc(r.family)}}</td>`
    + `<td class="name">${{esc(r.name)}}</td>`
    + `<td class="file">${{esc(r.file)}}</td>`
    + `<td>${{r.line}}</td>`
    + `<td class="reason">${{esc(r.reason)}}</td></tr>`
  ).join('') + (view.length > MAX
    ? `<tr><td colspan="7" class="count">… ${{view.length - MAX}} more (narrow the filter)</td></tr>` : '');
}}

document.querySelectorAll('th[data-k]').forEach(th => th.addEventListener('click', () => {{
  const k = th.dataset.k;
  if (k === sortKey) sortDir = -sortDir; else {{ sortKey = k; sortDir = 1; }}
  render();
}}));
[q, kindSel, axisSel, famSel].forEach(el => el.addEventListener('input', render));
render();
</script>
</body>
</html>
"""


def render(data: dict, label: str = "") -> str:
    # Escape "</" so an embedded string can't close the <script> early.
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return _TEMPLATE.format(data=blob, label=label or "n64decomp/sm64")
