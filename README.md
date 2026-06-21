# sm64_doc_measurer

A tool to measure the documentation completeness of the decompiled
[SM64 source](https://github.com/n64decomp/sm64).

Every named symbol the decomp exposes — functions, their arguments and local
variables, structs and their members, global variables, `#define` constants, enum
members, and object fields (`#define oFoo OBJECT_FIELD_*(…)`) — is classified as
**GOOD**, **MALFORMED** (named but off-convention, e.g. a leading capital or an
embedded address), or **UNDOCUMENTED** (a placeholder name like `func_8024…`,
`D_8033…`, `unk02`, `oUnk94`, `ACT_UNKNOWN_5`). Those classifications roll up into a
per-category completeness score, and — more usefully — into a worklist of exactly
what still needs a name.

**`unused` is not a defect.** A matching decompilation must keep unused content,
and naming it (`MODEL_DOOR_UNUSED`, `gUnusedLoadedPool`, `unused_ice_shard`) is
the *correct*, documented thing — it records real knowledge, unlike `unk`, which
records ignorance. Only the address-named ones (`sUnused80226B40`) stay flagged,
for the address.

## Current-state engine (`doc_measure.py`)

The recommended path. It parses the raw checkout with
[tree-sitter](https://tree-sitter.github.io/), so **no build of the decomp is
required** — point it at any clone and go.

```sh
pip install -r requirements.txt
python3 doc_measure.py /path/to/sm64 --json out.json --html site/index.html
```

It prints a per-category completeness table and then *what needs attention
right now* — the files with the most placeholder symbols, and a sample of the
offending names with `file:line` provenance:

```
category     documented   completeness
------------------------------------------------------
function     4333/4553  ███████████████████████░  95.2%
arg          4509/5461  ████████████████████░░░░  82.6%
local        5884/6786  █████████████████████░░░  86.7%
struct        240/256   ██████████████████████░░  93.8%
member       1693/1955  █████████████████████░░░  86.6%
global       3173/3836  ████████████████████░░░░  82.7%
constant     3704/3807  ███████████████████████░  97.3%
enum         1479/1529  ███████████████████████░  96.7%
object_field    524/712   ██████████████████░░░░░░  73.6%
------------------------------------------------------
OVERALL                 █████████████████████░░░  88.3%
```

`--json` writes the full result, including every symbol needing attention, for
feeding a dashboard or a "good first issue" list. `--html` writes a
self-contained report page (data embedded, no external assets) with both axes'
bars and a sortable/filterable worklist.

### Two axes: completeness and uniformity

Documentation has two goals, scored separately:

- **Completeness** (above) — does a symbol have a real, non-placeholder name?
- **Uniformity** — of the names that *are* real, do they follow the project's
  conventions? This is a second, *role-aware* layer over the layer-1 GOOD names:
  functions should be `snake_case`, types `PascalCase`, struct members
  `camelCase`, globals carry a `g`/`s` prefix, and a `BehaviorScript` global must
  be `bhv…` (a type-aware family rule), `#define` constants / enum members should
  be `UPPER_SNAKE`, object fields `o…` PascalCase, and an include guard (the
  `guard` kind) must end in `_H`. Snake_case data tables and `_`-prefixed linker
  symbols are accepted alternatives, not flagged. Two specific misspellings are
  called out by name: `beh_` is flagged as a typo for the behavior prefix `bhv_`.

```
convention   conforming   uniformity
------------------------------------------------------
function     4330/4333  ████████████████████████  99.9%
struct        237/240   ████████████████████████  98.8%
member       1691/1693  ████████████████████████  99.9%
global       2802/3173  █████████████████████░░░  88.3%
constant     3681/3704  ████████████████████████  99.4%
enum         1043/1479  █████████████████░░░░░░░  70.5%
object_field    524/524   ████████████████████████ 100.0%
guard         180/182   ████████████████████████  98.9%
------------------------------------------------------
OVERALL                 ███████████████████████░  94.5%
```

(The lower enum uniformity is the decomp's intentionally lowercase preset enums —
`special_*`, `macro_*` — which the `UPPER_SNAKE` rule flags by design. The `guard`
row catches real typos: `MARIO_ACTIONS_MOVING`/`_STATIONARY` opened
`#ifndef MARIO_ACTIONS_MOVING` instead of `…_MOVING_H`, breaking the house style.)

### Entity families

Beyond the C-syntactic kind, constants and enum members are grouped into
**families by prefix** — `ACT_*` (Mario actions), `SOUND_*`, `MODEL_*`,
`DIALOG_*`, `bhv*` (behaviors), … — so a domain entity like "Mario actions" is a
named, navigable thing rather than an anonymous slice of "constant". Families are
auto-discovered (any prefix with ≥10 members; the long tail collapses to
`(other)`), scored by completeness, and the worklist can be filtered to one
family. (Per-family *uniformity* is deliberately not reported: a family is a
prefix group, so "do its members share the prefix?" is true by construction.)

### Semantic entities

A prefix family answers "what's it called?"; a **semantic entity** answers "is it
wired up?" — by cross-referencing a symbol to its *implementation* across files.
This isn't tautological, because the check crosses kinds and representations (a C
enum vs. a data-table macro, a level constant vs. its script).

The **scoping rule** (learned the hard way) is to only check links the decomp is
*required* to have for a correct, matching build, where a missing counterpart is a
genuine gap — not merely-conventional, optional links. Every `ACT_`/`CUTSCENE_`
does **not** need its own handler/dispatch entry: handlers are routinely shared or
inlined (one `act_star_dance()` serves `ACT_STAR_DANCE_EXIT`/`_NO_EXIT`/…), and
that's how the original game is built. Those checks were tried and removed — "no
handler" was noise, not a defect. What remains is the required content:

- **Dialog** — a numbered `DIALOG_NNN` (in `dialog_ids.h`) must have on-screen
  text, a `DEFINE_DIALOG(DIALOG_NNN, …)` row in `text/us/dialogs.h`. 170/170.
- **Level** — a `LEVEL_X` (a `DEFINE_LEVEL` row in `levels/level_defines.h`)
  must have a `level_<folder>_entry[]` script in `levels/<folder>/script.c`.
  31/31. (Fully cross-file: the constant, the folder, and the script all differ.)
- **Music sequence** — a `SEQ_X` (in `seq_ids.h`) must have an actual `.m64` in
  the audio manifest `sound/sequences.json`. 35/35.

These roll up into a third headline number, **wired up** (linked / total across
all entities) — currently all green, so it doubles as a regression guard: add a
`DIALOG_` id with no text and it goes red. Each entity reads repo files, so it
only runs when pointed at a real checkout.

Gaps show up in the worklist tagged `S`; the JSON has `semantic_entities` (the
summary) and `semantic_findings` (the gaps).

Convention violations appear in the same worklist (tagged `C`, with the reason),
and in the JSON under `violations` / `conventions` / `uniformity_score`. Every
completeness row carries a specific, actionable **reason** — `auto func_<addr>`,
`raw stack slot (sp<offset>)`, `padding/filler`, … — not just "undocumented".

### The report page

The HTML is self-contained (data embedded, no external assets) and has a
**dark/light toggle** (it follows your OS by default and remembers your choice),
three headline scores, per-kind bars colored by health, the family and semantic
tables, and one sortable/filterable worklist with a tag legend (`U`/`M`/`C`/`S`).

It's built to stay short and lead with what needs attention: each section is a
**collapsible panel** that shows its headline stat when closed, the long family
table collapses by default and hides fully-complete families, and **every metric
is clickable** — a scorecard, a bar, a family, or an entity — to filter the
worklist to it and jump straight there.

### Documentation comments

Beyond names, there's *prose*: the decomp's house style is a `/** … */` block
above each function (pervasive in the behaviors). The report measures doc-comment
coverage **per source area** (`src/<area>/…`), because it varies wildly — `menu`
is ~92%, `audio` ~20% — and an aggregate would hide that. It's informational (not
every function needs prose), so it isn't folded into the completeness score; the
JSON carries it under `doc_comments`.

(An aside on what *didn't* make the cut. A "storage class must match prefix" rule
— `static`→`s`, `extern`→`g` — was prototyped and dropped: in SM64 the `s`/`g`
prefix documents *intent*, not the literal `static` keyword (`sAreaYaw` is
`s`-prefixed but has global linkage), so it would have produced 500+ false
positives. A "`const Gfx` display lists must contain a `dl` token" rule was also
dropped despite a strong base (97.6% of 3,759 display lists do): all 90 exceptions
live in a single file, `actors/mario/model.inc.c`, with clean semantic names like
`mario_torso` / `mario_left_arm`. That's one author's consistent style choice, not
90 defects — flagging it would be noise, the very thing the worklist exists to
avoid.)

### Hosted report (GitHub Pages)

`.github/workflows/pages.yml` checks out the latest `n64decomp/sm64`, runs the
measurer, and publishes the HTML report to GitHub Pages — on every push, weekly,
or on demand. To turn it on: **Settings → Pages → Build and deployment →
Source: GitHub Actions**.

### Notes / caveats

- The classifiers are name heuristics tuned to SM64 conventions; they aren't
  perfect, so treat the worklist as a strong signal, not gospel.
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
