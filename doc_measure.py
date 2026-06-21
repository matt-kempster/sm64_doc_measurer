#!/usr/bin/env python3
"""Measure the documentation completeness of a decompiled SM64 source tree.

This is the modernized engine. The original ``sm64_parse.py`` shelled out to
``cpp`` + ``pycparser``, which needs a *built* decomp (generated headers,
``NON_MATCHING``); this one parses the raw checkout with **tree-sitter**, so it
runs on any clone with no build step.

What it measures is unchanged in spirit: every named symbol the decomp exposes
-- functions, their args and locals, structs and their members, and global
variables -- is classified ``GOOD`` / ``MALFORMED`` / ``UNDOCUMENTED`` by the
same name heuristics, and aggregated into a per-category completeness score.

The output is reoriented toward *current state*: rather than a score-over-time
graph, it reports what still needs attention right now -- the offending symbols,
grouped by file -- so a contributor can see exactly what to fix.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import tree_sitter_c
from tree_sitter import Language, Parser

# --------------------------------------------------------------------------- #
# tree-sitter front end
# --------------------------------------------------------------------------- #

# Blank preprocessor conditional lines (keeping the newline so line numbers are
# preserved). tree-sitter parses C, not the preprocessor, so an "#if ... #endif"
# that splits a brace pair leaves the token stream unbalanced and the enclosing
# function fails to parse. Blanking the directive line -- but keeping the guarded
# code -- restores a parseable body. Same trick sm64.sql uses.
_PREPROC_COND = re.compile(
    rb"^[ \t]*#[ \t]*(?:if|ifdef|ifndef|elif|else|endif)\b.*$", re.MULTILINE
)

_parser: Optional[Parser] = None


def get_parser() -> Parser:
    global _parser
    if _parser is None:
        language = Language(tree_sitter_c.language(), "c")
        parser = Parser()
        parser.set_language(language)
        _parser = parser
    return _parser


def parse_source(src: bytes):
    blanked = _PREPROC_COND.sub(lambda m: b" " * len(m.group(0)), src)
    return get_parser().parse(blanked)


def _iter(node, node_type: str) -> Iterable:
    if node.type == node_type:
        yield node
    for child in node.children:
        yield from _iter(child, node_type)


def _descend_to_identifier(node) -> Optional[str]:
    """Find the declared name inside a (possibly nested) declarator."""
    if node is None:
        return None
    # "field_identifier" is how the grammar names a struct member.
    if node.type in ("identifier", "field_identifier"):
        return node.text.decode()
    # init_declarator / pointer_declarator / array_declarator / function_declarator
    # / parenthesized_declarator all wrap the real declarator under "declarator".
    inner = node.child_by_field_name("declarator")
    if inner is not None:
        return _descend_to_identifier(inner)
    for child in node.named_children:
        if child.type.endswith("declarator") or child.type in (
            "identifier",
            "field_identifier",
        ):
            name = _descend_to_identifier(child)
            if name is not None:
                return name
    return None


def function_name(node) -> Optional[str]:
    return _descend_to_identifier(node.child_by_field_name("declarator"))


def function_params(node) -> List[str]:
    declarator = node.child_by_field_name("declarator")
    fdecl = next(_iter(declarator, "function_declarator"), None) if declarator else None
    plist = fdecl.child_by_field_name("parameters") if fdecl is not None else None
    if plist is None:
        return []
    params: List[str] = []
    for child in plist.named_children:
        if child.type != "parameter_declaration":
            continue
        name = _descend_to_identifier(child.child_by_field_name("declarator"))
        if name is not None:
            params.append(name)
    return params


def _declaration_names(decl) -> List[str]:
    """Identifiers declared by a ``declaration`` node (handles ``int a, b;``)."""
    names: List[str] = []
    declarators = decl.children_by_field_name("declarator")
    if not declarators:  # bare ``struct Foo;`` etc.
        return names
    for d in declarators:
        name = _descend_to_identifier(d)
        if name is not None:
            names.append(name)
    return names


def _is_prototype(decl) -> bool:
    """True if a top-level declaration is a function prototype, not a variable."""
    for d in decl.children_by_field_name("declarator"):
        if next(_iter(d, "function_declarator"), None) is not None:
            return True
    return False


def _collect_members(body):
    """Yield (name, field_node) for every member declared in a struct/union body.

    Descends into anonymous nested unions/structs -- whose fields are accessed
    directly on the parent (e.g. the unnamed unions inside ``struct Object``), so
    they're real members that were previously dropped entirely."""
    for fd in body.named_children:
        if fd.type != "field_declaration":
            continue
        names = _declaration_names(fd)
        if names:
            for n in names:
                yield n, fd
        else:
            # An anonymous aggregate field (no member name): recurse into it.
            for c in fd.children:
                if c.type in ("struct_specifier", "union_specifier"):
                    inner = c.child_by_field_name("body")
                    if inner is not None:
                        yield from _collect_members(inner)


def _typedef_name(td) -> Optional[str]:
    """The new type name introduced by a ``type_definition`` (the typedef name).

    It's a ``type_identifier`` under the declarator; for pointer typedefs it's
    nested. Only look inside the declarator, never the struct body (whose member
    types are also type_identifiers)."""
    d = td.child_by_field_name("declarator")
    if d is not None:
        if d.type == "type_identifier":
            return d.text.decode()
        ti = next(_iter(d, "type_identifier"), None)
        return ti.text.decode() if ti is not None else None
    tids = [c for c in td.children if c.type == "type_identifier"]
    return tids[-1].text.decode() if tids else None


# --------------------------------------------------------------------------- #
# Classification (ported verbatim from the original sm64_parse.py heuristics)
# --------------------------------------------------------------------------- #


class Classification(Enum):
    UNDOCUMENTED = auto()
    MALFORMED = auto()
    GOOD = auto()


def classify_function_name(name: str) -> Classification:
    lower = name.lower()
    if any(lower.startswith(prefix) for prefix in ["func", "unk", "proc8"]):
        return Classification.UNDOCUMENTED
    if lower != name or ("80" in name and not name.startswith("approach")):
        return Classification.MALFORMED
    return Classification.GOOD


def classify_struct_name(name: str) -> Classification:
    prefixes = ["dummy", "struct", "substruct"]
    lower = name.lower()
    if (
        any(lower.startswith(prefix) for prefix in prefixes)
        or lower.endswith("sub")
        or re.match("GraphNode[_0-9]+", name)
        or "thing" in lower
        or "unk" in lower
    ):
        return Classification.UNDOCUMENTED
    return Classification.GOOD


def classify_arg(arg: str) -> Classification:
    if (
        # Stack-slot names: "sp" + hex offset. Use the same regex (and the
        # space/speed exception) as classify_local_var, rather than the original
        # "starts with sp and short", which flagged real words like spawn/space.
        (re.match(r"sp[0-9A-Fa-f]+$", arg) and arg not in ["space", "speed"])
        or arg.startswith("arg")
        or (arg.startswith("a") and len(arg) <= 2)
        or (len(arg) == 1 and arg not in ["m", "x", "y", "z"])
    ):
        return Classification.UNDOCUMENTED
    if "_" in arg:
        return Classification.MALFORMED
    return Classification.GOOD


def classify_struct_member(name: str) -> Classification:
    if (
        name.startswith("unk")
        or name.startswith("u_")
        or name.startswith("d_")
        or name == "plane28"
    ):
        return Classification.UNDOCUMENTED
    if name[0].isupper() or name.startswith("filler") or name.startswith("pad"):
        return Classification.MALFORMED
    return Classification.GOOD


def classify_local_var(name: str) -> Classification:
    if (
        (re.match(r"sp[0-9A-Fa-f]+$", name) and name not in ["space", "speed"])
        or name.startswith("unk")
        or re.match(r"val[0-9A-Fa-f]*$", name)
        or re.match(r"[abf][0-9]+$", name)
        or re.match(r"arg[0-9].*", name)
    ):
        return Classification.UNDOCUMENTED
    if name.startswith("pad") or name.startswith("filler"):
        return Classification.MALFORMED
    return Classification.GOOD


def classify_global_var(name: str) -> Classification:
    if (
        name.startswith("D_")
        or (
            re.match(r".*[0-9A-Fa-f]{5,}.*", name)
            and name != "sBowserPuzzlePieceActions"
        )
        or (name.startswith("bhv") and name[-1].isdigit())
    ):
        return Classification.UNDOCUMENTED
    # "unused" is not a defect (a matching decomp must keep unused data, and
    # naming it ``unused_ice_shard`` / ``gUnusedLoadedPool`` is correct). The
    # address-named ones (``sUnused80226B40``) are already caught above by the hex
    # rule, so only the casing convention matters here.
    if name[0].isupper():
        return Classification.MALFORMED
    return Classification.GOOD


def classify_constant(name: str) -> Classification:
    """Completeness for #define constants and enum members (placeholder check).

    Casing is checked separately on the uniformity axis; here we only ask whether
    the name is a placeholder: an unknown marker, a D_-name, or a trailing hex
    address (a run of >=4 hex digits, with at least one numeral so real words like
    SURFACE/DEFAULT aren't caught).

    NOTE: "unused" is deliberately *not* a defect. In a matching decompilation,
    unused content is required to stay, and naming it ``..._UNUSED`` (e.g.
    ``MODEL_DOOR_UNUSED``) is the documented, correct thing to do -- it records
    real knowledge ("this is unused"), unlike ``UNK``, which records ignorance.
    Only the address tail (``UNUSED_COUNT_80333EE8``) is still flagged, as a name
    that hasn't actually been figured out.
    """
    upper = name.upper()
    if "UNK" in upper or name.startswith("D_"):
        return Classification.UNDOCUMENTED
    tail = name.rsplit("_", 1)[-1]
    if re.fullmatch(r"[0-9A-Fa-f]{4,}", tail) and any(c.isdigit() for c in tail):
        return Classification.UNDOCUMENTED
    return Classification.GOOD


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

# The symbol categories, each with its classifier. "kind" is the label shown in
# the report; classifiers take a bare name and return a Classification.
CLASSIFIERS = {
    "function": classify_function_name,
    "arg": classify_arg,
    "local": classify_local_var,
    "struct": classify_struct_name,
    "member": classify_struct_member,
    "global": classify_global_var,
    "constant": classify_constant,  # object-like #define
    "enum": classify_constant,  # enum member
    "object_field": classify_constant,  # #define oFoo OBJECT_FIELD_*(...)
}


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str  # one of CLASSIFIERS
    classification: Classification
    file: str  # repo-relative
    line: int  # 1-based
    type_name: Optional[str] = None  # declared C type (globals), for role-aware rules


def _classify(name: str, kind: str) -> Classification:
    return CLASSIFIERS[kind](name)


# --------------------------------------------------------------------------- #
# Convention / uniformity layer (layer 2)
# --------------------------------------------------------------------------- #
#
# Layer 1 (the classifiers above) answers "is this a real, non-placeholder
# name?". Layer 2 asks a *role-aware* question of the names that pass: "does it
# follow the project's conventions?" It is scored on its own axis, over only the
# layer-1 GOOD names, so the two goals -- completeness and uniformity -- stay
# legible separately.

# Kinds that carry a convention rule (args/locals have no settled casing rule).
CONVENTION_KINDS = (
    "function",
    "struct",
    "member",
    "global",
    "constant",
    "enum",
    "object_field",
)

_SNAKE = re.compile(r"[a-z][a-z0-9_]*$")  # set_mario_action, dialog_table_eu_en
_PASCAL = re.compile(r"[A-Z][A-Za-z0-9]*$")  # MarioState
_CAMEL = re.compile(r"[a-z][A-Za-z0-9]*$")  # rawStickX
_UPPER_SNAKE = re.compile(r"[A-Z][A-Z0-9_]*$")  # ACT_WALKING, SOUND_GENERAL_COIN
_BHV = re.compile(r"bhv[A-Z]")  # bhvStarDoor
_GLOBAL_PREFIX = re.compile(r"(?:[gs]|gd|bhv)[A-Z]")  # gMarioState, sCount, gdFoo
_OBJ_FIELD = re.compile(r"o[A-Z][A-Za-z0-9]*$")  # oAction, oIntangibleTimer


def completeness_reason(sym: Symbol) -> str:
    """A specific, actionable reason a symbol is flagged on the completeness axis.

    The classifiers return only GOOD/MALFORMED/UNDOCUMENTED; this names the actual
    pattern -- "auto func_<addr>", "unk field", "sp<offset> stack slot" -- so the
    worklist tells a contributor *what to do*, not just that something is wrong.
    """
    if sym.classification == Classification.GOOD:
        return ""
    name = sym.name
    low = name.lower()
    if sym.classification == Classification.UNDOCUMENTED:
        if low.startswith("func") or low.startswith("proc8"):
            return "auto-generated name (func_<addr>) — give it a real name"
        if name.startswith("D_"):
            return "auto-generated data symbol (D_<addr>) — give it a real name"
        if low.startswith("unk") or low.startswith("u_") or low.startswith("d_"):
            return "unknown field (unk…) — identify and name it"
        if re.match(r"sp[0-9A-Fa-f]+$", name):
            return "raw stack slot (sp<offset>) — give it a real name"
        if re.search(r"[0-9A-Fa-f]{4,}", name):
            return "embedded ROM address in name — give it a real name"
        if name.startswith("arg") or re.fullmatch(r"a[0-9]", name):
            return "positional placeholder (arg<n>) — name the parameter"
        if re.fullmatch(r"[abf][0-9]+", name) or re.match(r"val[0-9A-Fa-f]*$", name):
            return "register/temp placeholder — give it a real name"
        if len(name) == 1:
            return "single-letter placeholder — give it a descriptive name"
        return "placeholder name — give it a real name"
    # MALFORMED: a real-ish name that breaks the basic shape for its kind.
    if name[:1].isupper():
        return "off-convention — leading uppercase"
    if low.startswith("filler") or low.startswith("pad") or name == "plane28":
        return "padding/filler field — name it or mark reserved"
    if "thing" in low:
        return "vague placeholder ('thing') — name it specifically"
    if re.search(r"[0-9A-Fa-f]{4,}", name) or "80" in name:
        return "embedded address in name — give it a real name"
    if "_" in name:
        return "off-convention — underscore where one word is expected"
    return "off-convention name"


def convention_violation(sym: Symbol) -> Optional[str]:
    """Return why a (layer-1 GOOD) symbol breaks convention, or None if it conforms."""
    name = sym.name
    if sym.kind == "function":
        if not _SNAKE.match(name):
            return "function should be snake_case"
    elif sym.kind == "struct":
        if not _PASCAL.match(name):
            return "type should be PascalCase"
    elif sym.kind == "member":
        if not _CAMEL.match(name):
            return "member should be camelCase"
    elif sym.kind == "global":
        # Type-aware family rule: a BehaviorScript must be bhv + PascalCase.
        if sym.type_name and "BehaviorScript" in sym.type_name:
            return None if _BHV.match(name) else "behavior should be bhv + PascalCase"
        # General globals: a g/s (extern/static) prefix is the convention.
        # snake_case data tables and _linker symbols are accepted alternatives.
        if _GLOBAL_PREFIX.match(name) or _SNAKE.match(name) or name.startswith("_"):
            return None
        return "global should have a g/s prefix"
    elif sym.kind in ("constant", "enum"):
        if not _UPPER_SNAKE.match(name):
            return "constant should be UPPER_SNAKE"
    elif sym.kind == "object_field":
        # Object fields are #define oFoo OBJECT_FIELD_*(...) -- intentionally
        # o + PascalCase, NOT UPPER_SNAKE (so they must not be lumped with
        # constants, or all ~700 would read as violations).
        if not _OBJ_FIELD.match(name):
            return "object field should be o + PascalCase"
    return None


# --------------------------------------------------------------------------- #
# Entity families (semantic grouping by prefix)
# --------------------------------------------------------------------------- #
#
# Beyond the C-syntactic kind, constants and enum members cluster into semantic
# families by shared prefix -- ACT_* (Mario actions), SOUND_*, MODEL_*, ... --
# and behaviors form their own family. Surfacing these makes "ACT_" a navigable
# entity rather than an anonymous member of "constant".

MIN_FAMILY_MEMBERS = 10  # smaller prefix groups collapse into "(other)"
OTHER_FAMILY = "(other)"


def family_of(sym: Symbol) -> Optional[str]:
    """The semantic family a symbol belongs to, or None if it isn't family-grouped."""
    if sym.kind == "global" and sym.type_name and "BehaviorScript" in sym.type_name:
        return "bhv*"
    if sym.kind in ("constant", "enum"):
        prefix = sym.name.split("_", 1)[0]
        return prefix + "_*" if prefix else None
    return None


def _named_families(symbols: List[Symbol], min_members: int) -> set:
    """Family labels with enough members to be worth naming (rest -> OTHER_FAMILY)."""
    sizes: Dict[str, int] = defaultdict(int)
    for s in symbols:
        fam = family_of(s)
        if fam:
            sizes[fam] += 1
    return {f for f, n in sizes.items() if n >= min_members}


def family_label(sym: Symbol, named: set) -> Optional[str]:
    fam = family_of(sym)
    if fam is None:
        return None
    return fam if fam in named else OTHER_FAMILY


def family_counts(
    symbols: List[Symbol], min_members: int = MIN_FAMILY_MEMBERS
) -> Dict[str, Dict[str, int]]:
    """Per-family tallies: count and complete (GOOD).

    Note we deliberately do NOT report per-family *uniformity*: a family is a
    prefix group, so "do its members share the prefix/casing?" is true by
    construction -- a tautology. Convention is measured per-kind (above) and
    semantics per-entity (below), where the question isn't circular.
    """
    named = _named_families(symbols, min_members)
    out: Dict[str, Dict[str, int]] = {}
    for s in symbols:
        label = family_label(s, named)
        if label is None:
            continue
        d = out.setdefault(label, {"count": 0, "good": 0})
        d["count"] += 1
        if s.classification == Classification.GOOD:
            d["good"] += 1
    return out


# --------------------------------------------------------------------------- #
# Semantic entities (cross-references, not just prefixes)
# --------------------------------------------------------------------------- #
#
# A prefix family answers "what's it called?"; a *semantic* entity answers "is it
# wired up?" -- by relating a symbol to its implementation across files. The check
# is not tautological (it crosses kinds and representations: a C enum to a
# data-table macro, a level constant to its script).
#
# IMPORTANT scoping rule, learned the hard way: only check links the decomp is
# *required* to have for a correct, matching build -- text for every dialog, a
# script for every level, audio for every sequence. A *missing* counterpart there
# is a genuine gap. Do NOT check merely-conventional, optional links: every
# ACT_/CUTSCENE_ does not need its own handler/dispatch entry (handlers are often
# shared or inlined, and that's how the original game is built), so "no handler"
# is not a defect -- it's noise. Those checks were tried and removed.


@dataclass(frozen=True)
class SemanticFinding:
    entity: str  # e.g. "Dialog"
    name: str  # the member missing its link
    detail: str  # what's missing
    file: str
    line: int


@dataclass
class SemanticContext:
    """What a semantic check sees: the parsed symbols, plus (optionally) the repo
    root so a check can read other representations -- text tables, level scripts,
    audio manifests -- that aren't part of the C symbol set."""

    symbols: List[Symbol]
    root: Optional[Path] = None

    def read(self, rel: str) -> Optional[str]:
        """Read a repo-relative file, or None if there's no root / it's missing."""
        if self.root is None:
            return None
        try:
            return (self.root / rel).read_text(errors="replace")
        except OSError:
            return None


def _entity_dialogs(ctx: SemanticContext):
    """A numbered dialog ID (DIALOG_NNN) should have on-screen text.

    The enum lives in include/dialog_ids.h; the text is a DEFINE_DIALOG(DIALOG_NNN,
    ...) row in text/us/dialogs.h. A missing entry is a dialog that's declared but
    never written -- the kind of gap this tool exists to surface."""
    text = ctx.read("text/us/dialogs.h")
    if text is None:
        return None
    defined = set(re.findall(r"DEFINE_DIALOG\(\s*(DIALOG_\w+)", text))
    ids = [
        s
        for s in ctx.symbols
        if s.kind == "enum"
        and s.file.endswith("dialog_ids.h")
        and re.fullmatch(r"DIALOG_\d+", s.name)
    ]
    findings = [
        SemanticFinding("Dialog", s.name, "no DEFINE_DIALOG text entry", s.file, s.line)
        for s in ids
        if s.name not in defined
    ]
    return _summary("Dialog", "DIALOG_NNN ⟷ on-screen text", ids, findings), findings


def _entity_sequences(ctx: SemanticContext):
    """A music sequence (SEQ_X) should have an actual .m64 in the audio manifest.

    The enum is include/seq_ids.h; the data is a key "<hex>_<name>" in
    sound/sequences.json. The enum keeps an EVENT_ infix the manifest drops, so we
    compare on the name suffix with that infix optionally stripped."""
    manifest = ctx.read("sound/sequences.json")
    if manifest is None:
        return None
    # Manifest keys look like "03_level_grass"; keep the part after the hex index.
    keys = {
        k.split("_", 1)[1]
        for k in re.findall(r'"([0-9A-Fa-f]{2}_[a-z0-9_]+)"', manifest)
    }
    seqs = [
        s
        for s in ctx.symbols
        if s.kind == "enum"
        and s.file.endswith("seq_ids.h")
        and s.name.startswith("SEQ_")
        and s.name not in ("SEQ_COUNT",)
    ]
    findings = []
    for s in seqs:
        suffix = s.name[len("SEQ_") :].lower()
        if suffix in keys or suffix.replace("event_", "", 1) in keys:
            continue
        findings.append(
            SemanticFinding(
                "Music sequence", s.name, "no .m64 in sequences.json", s.file, s.line
            )
        )
    return _summary("Music sequence", "SEQ_X ⟷ .m64 sequence", seqs, findings), findings


# Matches a DEFINE_LEVEL("name", LEVEL_X, COURSE_X, folder, ...) row; captures the
# level constant and the source-folder shorthand (4th arg) that names its script.
_DEFINE_LEVEL = re.compile(
    r"DEFINE_LEVEL\(\s*\"[^\"]*\"\s*,\s*(\w+)\s*,\s*\w+\s*,\s*(\w+)"
)


def _entity_levels(ctx: SemanticContext):
    """A level (LEVEL_X) should have a level-script entry point.

    levels/level_defines.h has one DEFINE_LEVEL row per real level, naming a source
    folder; the entry point is level_<folder>_entry[] in levels/<folder>/script.c.
    (STUB_LEVEL rows have no folder and are intentionally excluded.) This is a
    fully cross-file link: the constant, the folder, and the script all differ."""
    defines = ctx.read("levels/level_defines.h")
    if defines is None:
        return None
    rows = []  # (level_const, folder, line)
    for i, line in enumerate(defines.splitlines(), start=1):
        m = _DEFINE_LEVEL.search(line)
        if m:
            rows.append((m.group(1), m.group(2), i))
    findings = []
    for level, folder, line in rows:
        script = ctx.read(f"levels/{folder}/script.c")
        if not script or f"level_{folder}_entry" not in script:
            findings.append(
                SemanticFinding(
                    "Level",
                    level,
                    f"no level_{folder}_entry[] script",
                    "levels/level_defines.h",
                    line,
                )
            )
    return _summary("Level", "LEVEL_X ⟷ script entry point", rows, findings), findings


def _summary(entity: str, link: str, members, findings) -> dict:
    n = len(members)
    return {
        "entity": entity,
        "members": n,
        "linked": n - len(findings),
        "gaps": len(findings),
        "link": link,
    }


# Each entry: (SemanticContext) -> (summary dict, [SemanticFinding]) or None when
# the check can't run (e.g. a file-based check with no repo root available).
SEMANTIC_ENTITIES = [
    _entity_dialogs,
    _entity_levels,
    _entity_sequences,
]


def semantic_report(symbols: List[Symbol], root: Optional[Path] = None):
    ctx = SemanticContext(symbols, root)
    summaries = []
    findings: List[SemanticFinding] = []
    for check in SEMANTIC_ENTITIES:
        result = check(ctx)
        if result is None:
            continue
        summary, found = result
        summaries.append(summary)
        findings.extend(found)
    return summaries, findings


def extract_file(path: Path, rel: str) -> List[Symbol]:
    """Parse one C/H file and return every classified named symbol in it."""
    return extract_source(path.read_bytes(), rel)


def extract_source(src: bytes, rel: str) -> List[Symbol]:
    """Parse C source bytes and return every classified named symbol in it."""
    tree = parse_source(src)
    root = tree.root_node
    symbols: List[Symbol] = []

    def add(name: Optional[str], kind: str, node, type_name: Optional[str] = None):
        if not name:
            return
        symbols.append(
            Symbol(
                name,
                kind,
                _classify(name, kind),
                rel,
                node.start_point[0] + 1,
                type_name,
            )
        )

    # Functions, their args, and their locals.
    for fn in _iter(root, "function_definition"):
        fname = function_name(fn)
        # A function-like macro in declarator position (e.g. BAD_RETURN(s32))
        # parses as a "function" named in ALL_CAPS. Real sm64 functions are
        # snake_case, never all-caps, so skip those parse artifacts.
        if fname and re.fullmatch(r"[A-Z][A-Z0-9_]*", fname):
            fname = None
        add(fname, "function", fn)
        for arg in function_params(fn):
            add(arg, "arg", fn)
        body = fn.child_by_field_name("body")
        if body is not None:
            for decl in _iter(body, "declaration"):
                for name in _declaration_names(decl):
                    add(name, "local", decl)

    # Structs and their members, in both forms the decomp uses:
    #   struct Foo { ... };               (named struct_specifier)
    #   typedef struct { ... } Foo;       (anonymous struct, name from the typedef)
    # The second is the bulk of SM64's types; missing it under-counted structs and
    # all their members. Skip Dummy* (bss-reorder padding structs).
    def add_struct(sname: Optional[str], struct_node):
        body = struct_node.child_by_field_name("body")
        if not sname or body is None or re.match(r"Dummy[0-9]+$", sname):
            return
        members = list(_collect_members(body))
        if not members:  # forward-ish decl with no named fields; nothing to score
            return
        add(sname, "struct", struct_node)
        for mname, fd in members:
            add(mname, "member", fd)

    for st in _iter(root, "struct_specifier"):
        name_node = st.child_by_field_name("name")
        if name_node is not None:
            add_struct(name_node.text.decode(), st)

    # Anonymous typedef'd structs: the struct_specifier has no name, so take the
    # type name from the typedef's declarator. (Named typedefs like
    # `typedef struct Foo {} Foo;` are already handled by the loop above.)
    for td in _iter(root, "type_definition"):
        struct = next((c for c in td.children if c.type == "struct_specifier"), None)
        if struct is None or struct.child_by_field_name("name") is not None:
            continue
        add_struct(_typedef_name(td), struct)

    # Global variables: top-level declarations that aren't prototypes. Capture
    # the declared type so role-aware rules can fire (e.g. BehaviorScript).
    for decl in root.named_children:
        if decl.type != "declaration" or _is_prototype(decl):
            continue
        type_node = decl.child_by_field_name("type")
        type_name = type_node.text.decode() if type_node is not None else None
        for name in _declaration_names(decl):
            add(name, "global", decl, type_name)

    # Object-like #define constants. Skip include guards (they aren't real
    # constants): names ending in _H / _H_, or wrapped in underscores (_SM64_H_).
    # Object fields (`#define oFoo OBJECT_FIELD_*(...)`) get their own kind: they
    # are intentionally o+PascalCase, not UPPER_SNAKE, so scoring them as constants
    # would (wrongly) read all ~700 as convention violations.
    for d in _iter(root, "preproc_def"):
        nm = d.child_by_field_name("name")
        if nm is None:
            continue
        name = nm.text.decode()
        if _is_include_guard(name):
            continue
        kind = "object_field" if re.match(r"o[A-Z]", name) else "constant"
        add(name, kind, d)

    # Enum members.
    for e in _iter(root, "enumerator"):
        nm = e.child_by_field_name("name")
        if nm is not None:
            add(nm.text.decode(), "enum", e)

    return symbols


def _is_include_guard(name: str) -> bool:
    return bool(
        re.search(r"_H_?$", name)  # SM64_H, MACROS_H_
        or (name.startswith("_") and name.endswith("_"))  # _SM64_H_
        or name.endswith("_GUARD")
    )


def should_ignore_file(rel: str) -> bool:
    # The PR/ directory holds pre-decomp scratch; not real source.
    return "/PR/" in rel or rel.startswith("PR/")


def collect_symbols(root: Path) -> List[Symbol]:
    files = sorted(
        {
            *(root / "include").glob("*.h"),
            *(root / "src").glob("**/*.h"),
            *(root / "src").glob("**/*.c"),
        }
    )
    symbols: List[Symbol] = []
    for path in files:
        rel = str(path.relative_to(root))
        if should_ignore_file(rel):
            continue
        symbols.extend(extract_file(path, rel))
    return symbols


def _has_doc_comment(fn) -> bool:
    """True if a ``/** ... */`` block comment immediately precedes a function."""
    prev = fn.prev_sibling
    return prev is not None and prev.type == "comment" and prev.text.startswith(b"/**")


def doc_comment_coverage(root: Path) -> dict:
    """Per-area count of functions carrying a ``/** ... */`` doc comment.

    A direct *prose*-documentation measure, distinct from naming: the decomp's
    house style (seen throughout the behaviors) is a ``/**`` block above each
    function. Reported per source area (src/<area>/...) because coverage varies
    wildly -- menu is near-complete, audio is sparse -- and an aggregate hides
    that. Informational: not every function needs prose, so this is not folded
    into the completeness score."""
    areas: Dict[str, List[int]] = {}  # area -> [functions, documented]
    for path in sorted((root / "src").glob("**/*.c")):
        rel = str(path.relative_to(root))
        if should_ignore_file(rel):
            continue
        parts = rel.split("/")
        area = parts[1] if len(parts) > 2 else "src"
        tree = parse_source(path.read_bytes())
        a = areas.setdefault(area, [0, 0])
        for fn in _iter(tree.root_node, "function_definition"):
            a[0] += 1
            if _has_doc_comment(fn):
                a[1] += 1
    return {
        "total": {
            "functions": sum(v[0] for v in areas.values()),
            "documented": sum(v[1] for v in areas.values()),
        },
        "by_area": {
            k: {"functions": v[0], "documented": v[1]} for k, v in sorted(areas.items())
        },
    }


# --------------------------------------------------------------------------- #
# Scoring + reporting
# --------------------------------------------------------------------------- #


def category_counts(symbols: List[Symbol]) -> Dict[str, Dict[Classification, int]]:
    counts: Dict[str, Dict[Classification, int]] = {
        kind: defaultdict(int) for kind in CLASSIFIERS
    }
    for s in symbols:
        counts[s.kind][s.classification] += 1
    return counts


def overall_score(counts: Dict[str, Dict[Classification, int]]) -> float:
    ratios: List[float] = []
    for kind in CLASSIFIERS:
        c = counts[kind]
        total = sum(c.values())
        if total:
            ratios.append(c[Classification.GOOD] / total)
    return sum(ratios) / len(ratios) if ratios else 1.0


def uniformity_counts(symbols: List[Symbol]) -> Dict[str, Dict[str, int]]:
    """Per-kind conforming/violation tallies over layer-1 GOOD names only."""
    counts: Dict[str, Dict[str, int]] = {
        kind: {"CONFORMING": 0, "VIOLATION": 0} for kind in CONVENTION_KINDS
    }
    for s in symbols:
        if s.classification != Classification.GOOD or s.kind not in counts:
            continue
        key = "VIOLATION" if convention_violation(s) else "CONFORMING"
        counts[s.kind][key] += 1
    return counts


def uniformity_score(counts: Dict[str, Dict[str, int]]) -> float:
    conforming = sum(c["CONFORMING"] for c in counts.values())
    total = conforming + sum(c["VIOLATION"] for c in counts.values())
    return conforming / total if total else 1.0


def _bar(ratio: float, width: int = 24) -> str:
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


def print_report(
    symbols: List[Symbol], top_files: int, samples: int, root: Optional[Path] = None
) -> Tuple[float, float]:
    counts = category_counts(symbols)
    score = overall_score(counts)

    print(f"\nParsed {len(symbols)} named symbols.\n")
    print(f"{'category':<10} {'documented':>12}   completeness")
    print("-" * 54)
    for kind in CLASSIFIERS:
        c = counts[kind]
        total = sum(c.values())
        if not total:
            print(f"{kind:<10} {'(none)':>12}")
            continue
        good = c[Classification.GOOD]
        ratio = good / total
        print(f"{kind:<10} {good:>6}/{total:<5} {_bar(ratio)} {ratio * 100:5.1f}%")
    print("-" * 54)
    print(f"{'OVERALL':<10} {'':>12} {_bar(score)} {score * 100:5.1f}%\n")

    # Uniformity axis: of the names that pass completeness, which break convention?
    ucounts = uniformity_counts(symbols)
    uscore = uniformity_score(ucounts)
    print(f"{'convention':<10} {'conforming':>12}   uniformity")
    print("-" * 54)
    for kind in CONVENTION_KINDS:
        c = ucounts[kind]
        total = c["CONFORMING"] + c["VIOLATION"]
        if not total:
            print(f"{kind:<10} {'(none)':>12}")
            continue
        ratio = c["CONFORMING"] / total
        good = c["CONFORMING"]
        print(f"{kind:<10} {good:>6}/{total:<5} {_bar(ratio)} {ratio * 100:5.1f}%")
    print("-" * 54)
    print(f"{'OVERALL':<10} {'':>12} {_bar(uscore)} {uscore * 100:5.1f}%\n")

    problems = [s for s in symbols if s.classification != Classification.GOOD]
    if not problems:
        print("Nothing needs attention — fully documented. 🎉")
        return score, uscore

    # Files needing the most attention.
    by_file: Dict[str, int] = defaultdict(int)
    for s in problems:
        by_file[s.file] += 1
    print(f"Files needing the most attention ({len(by_file)} total):")
    for rel, n in sorted(by_file.items(), key=lambda kv: -kv[1])[:top_files]:
        print(f"  {n:>4}  {rel}")

    # A sample of offenders per category, so there's something concrete to grab.
    print("\nSample symbols needing attention (name @ file:line):")
    for kind in CLASSIFIERS:
        offenders = [s for s in problems if s.kind == kind]
        if not offenders:
            continue
        und = sum(
            1 for s in offenders if s.classification == Classification.UNDOCUMENTED
        )
        mal = len(offenders) - und
        print(f"\n  {kind} — {len(offenders)} ({und} undocumented, {mal} malformed):")
        for s in offenders[:samples]:
            tag = "U" if s.classification == Classification.UNDOCUMENTED else "M"
            print(
                f"    [{tag}] {s.name}  @ {s.file}:{s.line} — {completeness_reason(s)}"
            )
        if len(offenders) > samples:
            print(f"    … and {len(offenders) - samples} more")

    # A sample of convention (uniformity) violations.
    violations = [
        (s, convention_violation(s))
        for s in symbols
        if s.classification == Classification.GOOD
        and s.kind in CONVENTION_KINDS
        and convention_violation(s)
    ]
    if violations:
        print(f"\nConvention violations ({len(violations)} total):")
        for s, reason in violations[:samples]:
            print(f"    [C] {s.name}  @ {s.file}:{s.line} — {reason}")
        if len(violations) > samples:
            print(f"    … and {len(violations) - samples} more")

    # Entity families (constants/enums grouped by prefix) by completeness.
    fams = family_counts(symbols)
    if fams:
        print(f"\nEntity families ({len(fams)}), lowest completeness first:")
        ranked = sorted(fams.items(), key=lambda kv: kv[1]["good"] / kv[1]["count"])
        for fam, c in ranked[:top_files]:
            comp = 100 * c["good"] / c["count"]
            print(f"  {fam:<16} n={c['count']:<4} complete {comp:4.0f}%")

    # Semantic entities: cross-reference checks (not tautological).
    summaries, findings = semantic_report(symbols, root)
    if summaries:
        print("\nSemantic entities (implementation cross-references):")
        for sm in summaries:
            pct = 100 * sm["linked"] / sm["members"] if sm["members"] else 100.0
            print(
                f"  {sm['entity']:<16} {sm['link']}: "
                f"{sm['linked']}/{sm['members']} linked ({pct:.0f}%), {sm['gaps']} gaps"
            )
        for f in findings[:samples]:
            print(f"    [S] {f.name}  @ {f.file}:{f.line} — {f.detail}")
        if len(findings) > samples:
            print(f"    … and {len(findings) - samples} more")

    # Doc-comment coverage (prose docs), per source area.
    if root is not None:
        doc = doc_comment_coverage(root)
        t = doc["total"]
        if t["functions"]:
            pct = 100 * t["documented"] / t["functions"]
            print(
                f"\nDoc-comment coverage: {t['documented']}/{t['functions']} "
                f"functions ({pct:.0f}%), by area:"
            )
            for area, c in sorted(
                doc["by_area"].items(),
                key=lambda kv: (
                    kv[1]["documented"] / kv[1]["functions"]
                    if kv[1]["functions"]
                    else 1
                ),
            ):
                if c["functions"]:
                    p = 100 * c["documented"] / c["functions"]
                    print(
                        f"  {area:<10} {c['documented']:>4}/{c['functions']:<5}"
                        f" {_bar(p / 100)} {p:4.0f}%"
                    )
    return score, uscore


def to_json(symbols: List[Symbol], root: Optional[Path] = None) -> dict:
    counts = category_counts(symbols)
    ucounts = uniformity_counts(symbols)
    named = _named_families(symbols, MIN_FAMILY_MEMBERS)
    semantic_summaries, semantic_findings = semantic_report(symbols, root)
    return {
        "score": overall_score(counts),
        "categories": {
            kind: {cls.name: counts[kind][cls] for cls in Classification}
            for kind in CLASSIFIERS
        },
        "needs_attention": [
            {
                "name": s.name,
                "kind": s.kind,
                "family": family_label(s, named),
                "classification": s.classification.name,
                "reason": completeness_reason(s),
                "file": s.file,
                "line": s.line,
            }
            for s in symbols
            if s.classification != Classification.GOOD
        ],
        "uniformity_score": uniformity_score(ucounts),
        "conventions": ucounts,
        "violations": [
            {
                "name": s.name,
                "kind": s.kind,
                "family": family_label(s, named),
                "reason": convention_violation(s),
                "file": s.file,
                "line": s.line,
            }
            for s in symbols
            if s.classification == Classification.GOOD
            and s.kind in CONVENTION_KINDS
            and convention_violation(s)
        ],
        "families": family_counts(symbols),
        "semantic_entities": semantic_summaries,
        "semantic_findings": [
            {
                "entity": f.entity,
                "name": f.name,
                "detail": f.detail,
                "file": f.file,
                "line": f.line,
            }
            for f in semantic_findings
        ],
        "doc_comments": doc_comment_coverage(root) if root is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", metavar="PATH_TO_SM64_SOURCE_DIR", type=Path)
    parser.add_argument("--json", metavar="FILE", type=Path, help="write full results")
    parser.add_argument(
        "--html", metavar="FILE", type=Path, help="write a self-contained report page"
    )
    parser.add_argument("--label", default="", help="header text for the HTML report")
    parser.add_argument("--top-files", type=int, default=20)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    if not (args.root / "src").is_dir():
        print(
            f"error: {args.root}/src not found — is that a decomp checkout?",
            file=sys.stderr,
        )
        return 2

    symbols = collect_symbols(args.root)
    score, uscore = print_report(symbols, args.top_files, args.samples, args.root)
    data = to_json(symbols, args.root)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2))
        print(f"\nWrote {args.json}")
    if args.html:
        import report_html

        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(report_html.render(data, args.label))
        print(f"Wrote {args.html}")
    print(f"\ncompleteness: {score * 100:.4f}%   uniformity: {uscore * 100:.4f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
