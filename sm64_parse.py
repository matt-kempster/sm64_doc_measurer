#!/usr/bin/env python3.7

import itertools
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, NamedTuple, Set, Tuple

from pycparser import c_ast, parse_file
from pycparser.c_ast import FileAST, NodeVisitor

# File parsing


def get_ast_from_file(
    filename: Path, include_dirs: List[Path], force_includes: List[Path]
) -> FileAST:
    return parse_file(
        str(filename),
        use_cpp=True,
        cpp_args=[
            "-DVERSION_US",
            "-D_LANGUAGE_C",
            "-D__attribute__(x)=",
            "-D_Static_assert(a,b)=",
            "-D__builtin_va_list=char",
            "-D__builtin_va_arg(a,b)=1",
            *[f"-I{include_dir}" for include_dir in include_dirs],
            *[f"-include{force_include}" for force_include in force_includes],
        ],
    )


# Classes


class Classification(Enum):
    UNDOCUMENTED = auto()
    MALFORMED = auto()
    GOOD = auto()


@dataclass(frozen=True)
class ClassifiedSymbol:
    symbol_name: str
    classification: Classification


@dataclass(frozen=True)
class Function:
    name: ClassifiedSymbol
    args: Tuple[ClassifiedSymbol, ...]
    local_vars: Tuple[ClassifiedSymbol, ...]


@dataclass(frozen=True)
class Struct:
    name: ClassifiedSymbol
    members: Tuple[ClassifiedSymbol, ...]


@dataclass
class AllSymbols:
    functions: Set[Function] = field(default_factory=set)
    structs: Set[Struct] = field(default_factory=set)
    global_vars: Set[ClassifiedSymbol] = field(default_factory=set)


all_symbols = AllSymbols()


# Classification functions


def classify_function_name(name: str) -> Classification:
    lower = name.lower()
    if any(
        lower.startswith(prefix)
        for prefix in ["func", "unk", "proc8"]  # proc8 is a bit of a hack
    ):
        return Classification.UNDOCUMENTED
    if lower != name or "80" in name:  # no addresses allowed
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


def should_ignore_file(filename: str) -> bool:
    return Path(filename).parent.name == "PR"


def classify_arg(arg: str) -> Classification:
    if (
        (arg.startswith("sp") and len(arg) <= 5)
        or arg.startswith("arg")
        or (arg.startswith("a") and len(arg) <= 2)
        or (len(arg) == 1 and arg not in ["m", "x", "y", "z"])
    ):
        return Classification.UNDOCUMENTED
    if "_" in arg:
        return Classification.MALFORMED
    return Classification.GOOD


def classify_struct_member(name: str) -> Classification:
    if name.startswith("unk") or name.startswith("u_") or name.startswith("d_"):
        return Classification.UNDOCUMENTED
    if name.startswith("filler") or name.startswith("pad"):
        return Classification.MALFORMED
    return Classification.GOOD


def classify_local_var(name: str) -> Classification:
    return Classification.GOOD


def classify_all_args(node) -> Tuple[ClassifiedSymbol, ...]:
    ret: List[ClassifiedSymbol] = []
    try:
        args = [arg.name for arg in node.decl.type.args.params]
    except AttributeError:
        return tuple(ret)
    if not args:
        return tuple(ret)
    for arg in args:
        if isinstance(arg, str):  # sometimes can be None, idk
            ret.append(ClassifiedSymbol(arg, classify_arg(arg)))
    return tuple(ret)


def classify_all_local_vars(node) -> Tuple[ClassifiedSymbol, ...]:
    ret: List[ClassifiedSymbol] = []
    for _, block in node.body.children():
        if not isinstance(block, c_ast.Decl):
            return tuple(ret)  # decls come first according to c90
        ret.append(ClassifiedSymbol(block.name, classify_local_var(block.name)))
    return tuple(ret)


def classify_all_struct_members(node) -> Tuple[ClassifiedSymbol, ...]:
    ret: List[ClassifiedSymbol] = []
    if not node.decls or any(
        should_ignore_file(decl.coord.file) for decl in node.decls
    ):
        return tuple(ret)
    for decl in node.decls:
        ret.append(ClassifiedSymbol(decl.name, classify_struct_member(decl.name)))
    return tuple(ret)


def classify_global_var(name: str) -> Classification:
    if (
        name.startswith("D_")
        or re.match(r".*[0-9A-Fa-f]{5,}.*", name) is not None
        or (name.startswith("bhv") and name[-1].isdigit())
    ):
        return Classification.UNDOCUMENTED
    if name[0].isupper() or name.startswith("unused"):
        return Classification.MALFORMED
    return Classification.GOOD


# The visitors


class Visitor(NodeVisitor):
    def visit_FuncDef(self, node):
        global all_symbols
        name = node.decl.name
        all_symbols.functions.add(
            Function(
                name=ClassifiedSymbol(name, classify_function_name(name)),
                args=classify_all_args(node),
                local_vars=classify_all_local_vars(node),
            )
        )

    def visit_Struct(self, node):
        global all_symbols
        if not node.name:
            return
        struct = Struct(
            name=ClassifiedSymbol(node.name, classify_struct_name(node.name)),
            members=classify_all_struct_members(node),
        )
        all_symbols.structs.add(struct)


def collect_global_vars(ast: FileAST):
    global all_symbols

    for child in ast.children():
        decl = child[1]
        if isinstance(decl, c_ast.Decl) and decl.name:
            if isinstance(decl.children()[0][1], (c_ast.FuncDecl, c_ast.Struct)):
                continue
            all_symbols.global_vars.add(
                ClassifiedSymbol(decl.name, classify_global_var(decl.name))
            )


# Printing stuff


def print_classifications(classifications):
    for classification, names in classifications.items():
        if not names:
            continue
        print(f"{classification.name}: {', '.join(sorted(names))}\n")


def format_totals(classifications) -> str:
    summary = ""
    total = 0
    for classification, names in classifications.items():
        summary += f"{classification.name}={len(names)}, "
        total += len(names)
    summary += f"total={total}"
    return summary


def print_classified_symbols(symbols: List[ClassifiedSymbol]):
    d: Dict[Classification, List[str]] = defaultdict(list)
    for symbol in symbols:
        d[symbol.classification].append(symbol.symbol_name)
    print_classifications(d)
    print(format_totals(d))


def print_all_symbols(symbols: AllSymbols):
    print("FUNCTIONS")
    print_classified_symbols([function.name for function in symbols.functions])
    print("STRUCTS")
    print_classified_symbols([struct.name for struct in symbols.structs])
    print("GLOBAL VARS")
    print_classified_symbols([global_var for global_var in symbols.global_vars])

    print("STRUCT MEMBERS")
    print_classified_symbols(
        list(set(itertools.chain(*[struct.members for struct in symbols.structs])))
    )

    print("LOCAL VARS")
    print_classified_symbols(
        list(
            set(
                itertools.chain(
                    *[function.local_vars for function in symbols.functions]
                )
            )
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: sm64_parse PATH_TO_SM64_SOURCE_DIR")
        exit(1)

    root = Path(sys.argv[1])
    include_dir = root / "include"
    build_include_dir_jp = root / "build" / "jp" / "include"
    build_include_dir_us = root / "build" / "us" / "include"
    src_dir = root / "src"
    force_include_ultra64 = include_dir / "ultra64.h"
    force_include_sm64 = include_dir / "sm64.h"
    c_files = [
        *((root / "include").glob("*.h")),
        *((root / "src").glob("**/*.h")),
        *((root / "src").glob("**/*.c")),
    ]
    for i, filename in enumerate(c_files):
        if (i + 1) % 100 == 0:
            print(f"Done with {i + 1}...")

        ast = get_ast_from_file(
            filename,
            [include_dir, src_dir, build_include_dir_jp, build_include_dir_us],
            [force_include_sm64, force_include_ultra64],
        )
        collect_global_vars(ast)
        Visitor().visit(ast)

    print_all_symbols(all_symbols)

    breakpoint()
    exit(0)
