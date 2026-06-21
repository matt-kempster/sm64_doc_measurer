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
    if name[0].isupper() or name.startswith("unused"):
        return Classification.MALFORMED
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
CONVENTION_KINDS = ("function", "struct", "member", "global")

_SNAKE = re.compile(r"[a-z][a-z0-9_]*$")  # set_mario_action, dialog_table_eu_en
_PASCAL = re.compile(r"[A-Z][A-Za-z0-9]*$")  # MarioState
_CAMEL = re.compile(r"[a-z][A-Za-z0-9]*$")  # rawStickX
_BHV = re.compile(r"bhv[A-Z]")  # bhvStarDoor
_GLOBAL_PREFIX = re.compile(r"(?:[gs]|gd|bhv)[A-Z]")  # gMarioState, sCount, gdFoo


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
    return None


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

    # Structs and their members. Skip Dummy* (bss-reorder padding structs).
    for st in _iter(root, "struct_specifier"):
        name_node = st.child_by_field_name("name")
        body = st.child_by_field_name("body")
        if name_node is None or body is None:
            continue
        sname = name_node.text.decode()
        if re.match(r"Dummy[0-9]+$", sname):
            continue
        members = [
            n
            for fd in body.named_children
            if fd.type == "field_declaration"
            for n in _declaration_names(fd)
        ]
        if not members:  # forward-ish decl with no fields; nothing to score
            continue
        add(sname, "struct", st)
        for fd in body.named_children:
            if fd.type != "field_declaration":
                continue
            for mname in _declaration_names(fd):
                add(mname, "member", fd)

    # Global variables: top-level declarations that aren't prototypes. Capture
    # the declared type so role-aware rules can fire (e.g. BehaviorScript).
    for decl in root.named_children:
        if decl.type != "declaration" or _is_prototype(decl):
            continue
        type_node = decl.child_by_field_name("type")
        type_name = type_node.text.decode() if type_node is not None else None
        for name in _declaration_names(decl):
            add(name, "global", decl, type_name)

    return symbols


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
    symbols: List[Symbol], top_files: int, samples: int
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
            print(f"    [{tag}] {s.name}  @ {s.file}:{s.line}")
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
    return score, uscore


def to_json(symbols: List[Symbol]) -> dict:
    counts = category_counts(symbols)
    ucounts = uniformity_counts(symbols)
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
                "classification": s.classification.name,
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
                "reason": convention_violation(s),
                "file": s.file,
                "line": s.line,
            }
            for s in symbols
            if s.classification == Classification.GOOD
            and s.kind in CONVENTION_KINDS
            and convention_violation(s)
        ],
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
    score, uscore = print_report(symbols, args.top_files, args.samples)
    data = to_json(symbols)
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
