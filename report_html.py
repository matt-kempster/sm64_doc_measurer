"""Render a self-contained current-state HTML report.

One static file: the result JSON is embedded, and a little vanilla JS renders the
two axes -- completeness and uniformity -- as per-category bars plus a single
sortable/filterable worklist (undocumented + malformed + convention violations).
No external assets, so it drops straight onto GitHub Pages.
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
  .track {{ background: var(--line); border-radius: 6px; height: 14px; overflow: hidden; }}
  .fill {{ height: 100%; background: var(--good); }}
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
  td.name {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  td.file {{ color: var(--muted); font-family: ui-monospace, monospace; font-size: 12px; }}
  td.reason {{ color: var(--muted); font-size: 12px; }}
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

  <h2>Worklist</h2>
  <div class="controls">
    <input type="search" id="q" placeholder="filter by name or file…">
    <select id="axis">
      <option value="">both axes</option>
      <option value="completeness">completeness</option>
      <option value="convention">convention</option>
    </select>
    <select id="kind"></select>
    <span class="count" id="count"></span>
  </div>
  <table>
    <thead><tr>
      <th data-k="tag">!</th>
      <th data-k="kind">kind</th>
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

// One worklist combining both axes.
const tagReason = {{UNDOCUMENTED: 'undocumented', MALFORMED: 'malformed'}};
const rows = [
  ...DATA.needs_attention.map(r => ({{
    tag: r.classification === 'UNDOCUMENTED' ? 'U' : 'M', axis: 'completeness',
    kind: r.kind, name: r.name, file: r.file, line: r.line,
    reason: tagReason[r.classification] || '',
  }})),
  ...(DATA.violations || []).map(r => ({{
    tag: 'C', axis: 'convention',
    kind: r.kind, name: r.name, file: r.file, line: r.line, reason: r.reason,
  }})),
];

let sortKey = 'file', sortDir = 1;
const q = document.getElementById('q'), kindSel = document.getElementById('kind');
const axisSel = document.getElementById('axis');
const tbody = document.getElementById('rows'), countEl = document.getElementById('count');

const allKinds = [...new Set(rows.map(r => r.kind))].sort();
kindSel.innerHTML = '<option value="">all kinds</option>'
  + allKinds.map(k => `<option value="${{k}}">${{k}}</option>`).join('');

function esc(s) {{ return String(s).replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c])); }}

function render() {{
  const term = q.value.toLowerCase(), fk = kindSel.value, fa = axisSel.value;
  let view = rows.filter(r =>
    (!fk || r.kind === fk) && (!fa || r.axis === fa) &&
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
    + `<td class="name">${{esc(r.name)}}</td>`
    + `<td class="file">${{esc(r.file)}}</td>`
    + `<td>${{r.line}}</td>`
    + `<td class="reason">${{esc(r.reason)}}</td></tr>`
  ).join('') + (view.length > MAX
    ? `<tr><td colspan="6" class="count">… ${{view.length - MAX}} more (narrow the filter)</td></tr>` : '');
}}

document.querySelectorAll('th').forEach(th => th.addEventListener('click', () => {{
  const k = th.dataset.k;
  if (k === sortKey) sortDir = -sortDir; else {{ sortKey = k; sortDir = 1; }}
  render();
}}));
[q, kindSel, axisSel].forEach(el => el.addEventListener('input', render));
render();
</script>
</body>
</html>
"""


def render(data: dict, label: str = "") -> str:
    # Escape "</" so an embedded string can't close the <script> early.
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return _TEMPLATE.format(data=blob, label=label or "n64decomp/sm64")
