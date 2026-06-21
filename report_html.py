"""Render a self-contained current-state HTML report.

One static file: the result JSON is embedded, and a little vanilla JS renders the
per-category bars and a sortable/filterable worklist of every symbol that still
needs attention. No external assets, so it drops straight onto GitHub Pages.
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
    --good: #3fb950; --mal: #d29922; --und: #f85149; --line: #262a36;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  header {{ padding: 28px 24px 12px; }}
  h1 {{ margin: 0; font-size: 20px; }}
  .label {{ color: var(--muted); margin: 4px 0 0; font-size: 13px; }}
  .score {{ font-size: 56px; font-weight: 700; margin: 12px 0 0;
            font-variant-numeric: tabular-nums; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 0 24px 60px; }}
  .cats {{ display: grid; gap: 8px; margin: 18px 0 28px; }}
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
  .tag {{ display: inline-block; width: 1.6em; text-align: center; border-radius: 4px;
          font-weight: 700; font-size: 11px; padding: 1px 0; }}
  .tag.U {{ background: rgba(248,81,73,.18); color: var(--und); }}
  .tag.M {{ background: rgba(210,153,34,.18); color: var(--mal); }}
  .kind {{ color: var(--muted); }}
</style>
</head>
<body>
<header><div class="wrap" style="padding-bottom:0">
  <h1>SM64 Documentation — Current State</h1>
  <p class="label">{label}</p>
  <div class="score" id="score"></div>
</div></header>
<div class="wrap">
  <div class="cats" id="cats"></div>
  <div class="controls">
    <input type="search" id="q" placeholder="filter by name or file…">
    <select id="kind"></select>
    <select id="cls">
      <option value="">all problems</option>
      <option value="UNDOCUMENTED">undocumented</option>
      <option value="MALFORMED">malformed</option>
    </select>
    <span class="count" id="count"></span>
  </div>
  <table>
    <thead><tr>
      <th data-k="classification">!</th>
      <th data-k="kind">kind</th>
      <th data-k="name">name</th>
      <th data-k="file">file</th>
      <th data-k="line">line</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table>
</div>
<script>
const DATA = {data};
const KINDS = Object.keys(DATA.categories);

document.getElementById('score').textContent = (DATA.score * 100).toFixed(1) + '%';

// Category bars.
const cats = document.getElementById('cats');
for (const k of KINDS) {{
  const c = DATA.categories[k];
  const total = c.GOOD + c.MALFORMED + c.UNDOCUMENTED;
  const ratio = total ? c.GOOD / total : 1;
  const el = document.createElement('div');
  el.className = 'cat';
  el.innerHTML = `<span class="name">${{k}}</span>`
    + `<span class="track"><span class="fill" style="width:${{(ratio*100).toFixed(1)}}%"></span></span>`
    + `<span class="pct">${{(ratio*100).toFixed(1)}}% <small>${{c.GOOD}}/${{total}}</small></span>`;
  cats.appendChild(el);
}}

// Kind filter options.
const kindSel = document.getElementById('kind');
kindSel.innerHTML = '<option value="">all kinds</option>'
  + KINDS.map(k => `<option value="${{k}}">${{k}}</option>`).join('');

// Sortable, filterable table.
const rows = DATA.needs_attention;
let sortKey = 'file', sortDir = 1;
const q = document.getElementById('q'), kf = kindSel, cf = document.getElementById('cls');
const tbody = document.getElementById('rows'), countEl = document.getElementById('count');

function esc(s) {{ return String(s).replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c])); }}

function render() {{
  const term = q.value.toLowerCase(), fk = kf.value, fc = cf.value;
  let view = rows.filter(r =>
    (!fk || r.kind === fk) && (!fc || r.classification === fc) &&
    (!term || r.name.toLowerCase().includes(term) || r.file.toLowerCase().includes(term)));
  view.sort((a, b) => {{
    let x = a[sortKey], y = b[sortKey];
    return (x < y ? -1 : x > y ? 1 : 0) * sortDir;
  }});
  countEl.textContent = view.length.toLocaleString() + ' of '
    + rows.length.toLocaleString() + ' symbols need attention';
  const MAX = 2000;
  tbody.innerHTML = view.slice(0, MAX).map(r => {{
    const t = r.classification === 'UNDOCUMENTED' ? 'U' : 'M';
    return `<tr><td><span class="tag ${{t}}">${{t}}</span></td>`
      + `<td class="kind">${{r.kind}}</td>`
      + `<td class="name">${{esc(r.name)}}</td>`
      + `<td class="file">${{esc(r.file)}}</td>`
      + `<td>${{r.line}}</td></tr>`;
  }}).join('') + (view.length > MAX
    ? `<tr><td colspan="5" class="count">… ${{view.length - MAX}} more (narrow the filter)</td></tr>` : '');
}}

document.querySelectorAll('th').forEach(th => th.addEventListener('click', () => {{
  const k = th.dataset.k;
  if (k === sortKey) sortDir = -sortDir; else {{ sortKey = k; sortDir = 1; }}
  render();
}}));
[q, kf, cf].forEach(el => el.addEventListener('input', render));
render();
</script>
</body>
</html>
"""


def render(data: dict, label: str = "") -> str:
    # Escape "</" so an embedded string can't close the <script> early.
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return _TEMPLATE.format(data=blob, label=label or "n64decomp/sm64")
