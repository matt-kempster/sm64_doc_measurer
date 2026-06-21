# sm64_doc_measurer

A tool to measure the documentation completeness of the decompiled
[SM64 source](https://github.com/n64decomp/sm64).

Every named symbol the decomp exposes — functions, their arguments and local
variables, structs and their members, and global variables — is classified as
**GOOD**, **MALFORMED** (named but off-convention, e.g. a leading capital or an
embedded address), or **UNDOCUMENTED** (a placeholder name like `func_8024…`,
`D_8033…`, `unk02`). Those classifications roll up into a per-category
completeness score, and — more usefully — into a worklist of exactly what still
needs a name.

## Current-state engine (`doc_measure.py`)

The recommended path. It parses the raw checkout with
[tree-sitter](https://tree-sitter.github.io/), so **no build of the decomp is
required** — point it at any clone and go.

```sh
pip install -r requirements.txt
python3 doc_measure.py /path/to/sm64 --json out.json
```

It prints a per-category completeness table and then *what needs attention
right now* — the files with the most placeholder symbols, and a sample of the
offending names with `file:line` provenance:

```
category     documented   completeness
------------------------------------------------------
function     4333/4556  ███████████████████████░  95.1%
arg          4497/5461  ████████████████████░░░░  82.3%
local        5884/6786  █████████████████████░░░  86.7%
struct        238/254   ██████████████████████░░  93.7%
member       1688/1950  █████████████████████░░░  86.6%
global       3152/3836  ████████████████████░░░░  82.2%
------------------------------------------------------
OVERALL                 █████████████████████░░░  87.8%
```

`--json` writes the full result, including every symbol needing attention, for
feeding a dashboard or a "good first issue" list.

### Notes / caveats

- The classifiers are name heuristics ported verbatim from the original engine.
  They're tuned to SM64 conventions and aren't perfect — e.g. an argument named
  `speed` is flagged because the `sp`-prefix rule (stack pointers) doesn't carry
  the `space`/`speed` exception the local-variable rule has. Treat the worklist
  as a strong signal, not gospel.
- tree-sitter parses C, not the preprocessor: `#if VERSION_JP` / `#endif` lines
  are blanked (their guarded code is kept), so both version branches' symbols are
  counted. This slightly over-counts version-specific symbols — fine for a doc
  metric.

## Tests

```sh
python3 -m pytest
```

## Legacy engine (`sm64_parse.py` + the history graph)

The original engine used `pycparser` + `cpp` to get a fully preprocessed AST,
walked hundreds of commits, and rendered a score-over-time graph (jQuery/Flot via
`index.py` / `generate_site.py`) with a per-author "coin" leaderboard. The graph
is still here and still cool, but that path needs a **built** decomp (generated
headers like `levels/level_defines.h`, plus `NON_MATCHING`) and the heavier
`pycparser` / `htmldoom` deps — so it isn't wired into the current-state flow
above. See the commented-out section of `requirements.txt` to use it.
