#!/usr/bin/env python3.7

import argparse
import hashlib
import itertools
import pickle
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union

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
    filename: str
    args: Tuple[ClassifiedSymbol, ...]
    local_vars: Tuple[ClassifiedSymbol, ...]


@dataclass(frozen=True)
class Struct:
    name: ClassifiedSymbol
    filename: str
    members: Tuple[ClassifiedSymbol, ...]


@dataclass(frozen=True)
class GlobalVar:
    name: ClassifiedSymbol
    filename: str


@dataclass
class SymbolCollection:
    functions: Set[Function] = field(default_factory=set)
    structs: Set[Struct] = field(default_factory=set)
    global_vars: Set[GlobalVar] = field(default_factory=set)


@dataclass(frozen=True)
class Statistics:
    function_counts: Dict[Classification, int]
    struct_counts: Dict[Classification, int]
    global_var_counts: Dict[Classification, int]
    struct_member_counts: Dict[Classification, int]
    local_var_counts: Dict[Classification, int]


# Classification functions


def classify_function_name(name: str) -> Classification:
    lower = name.lower()
    if any(
        lower.startswith(prefix)
        for prefix in ["func", "unk", "proc8"]  # proc8 is a bit of a hack
    ):
        return Classification.UNDOCUMENTED
    if lower != name or (
        "80" in name and not name.startswith("approach")  # no addresses allowed
    ):
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


# The visitors


class FileVisitor(NodeVisitor):
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        self.functions: Set[Function] = set()
        self.structs: Set[Struct] = set()

    def visit_FuncDef(self, node):
        name = node.decl.name
        # if not name or should_ignore_file(node.coord.file):
        if not name or node.coord.file != self.filename:
            return
        self.functions.add(
            Function(
                name=ClassifiedSymbol(name, classify_function_name(name)),
                filename=node.coord.file,
                args=classify_all_args(node),
                local_vars=classify_all_local_vars(node),
            )
        )

    def visit_Struct(self, node):
        if (
            not node.name
            or node.coord.file != self.filename
            # or should_ignore_file(node.coord.file)
            # Suppress "Dummy" structs, which are used for bss reordering.
            or re.match(r"Dummy[0-9]+$", node.name)
        ):
            return
        struct = Struct(
            name=ClassifiedSymbol(node.name, classify_struct_name(node.name)),
            filename=node.coord.file,
            members=classify_all_struct_members(node),
        )
        if not struct.members:
            return
        self.structs.add(struct)


def collect_global_vars(ast: FileAST) -> Set[GlobalVar]:
    global_vars: Set[GlobalVar] = set()
    for child in ast.children():
        decl = child[1]
        if (
            not isinstance(decl, c_ast.Decl)
            or not decl.name
            or isinstance(decl.children()[0][1], (c_ast.FuncDecl, c_ast.Struct))
        ):
            continue
        global_vars.add(
            GlobalVar(
                name=ClassifiedSymbol(decl.name, classify_global_var(decl.name)),
                filename=decl.coord.file,
            )
        )
    return global_vars


# Counting stuff


def get_counts(symbols: List[ClassifiedSymbol]) -> Dict[Classification, int]:
    d: Dict[Classification, int] = defaultdict(int)
    for symbol in symbols:
        d[symbol.classification] += 1
    return d


def build_statistics(symbols: SymbolCollection) -> Statistics:
    return Statistics(
        function_counts=get_counts([function.name for function in symbols.functions]),
        struct_counts=get_counts([struct.name for struct in symbols.structs]),
        global_var_counts=get_counts(
            [global_var.name for global_var in symbols.global_vars]
        ),
        struct_member_counts=get_counts(
            list(itertools.chain(*[struct.members for struct in symbols.structs]))
        ),
        local_var_counts=get_counts(
            list(
                itertools.chain(
                    *[function.local_vars for function in symbols.functions]
                )
            )
        ),
    )


# Printing stuff


def print_classifications(classifications: Dict[Classification, List[str]]) -> None:
    for classification, names in classifications.items():
        if not names:
            continue
        print(f"{classification.name}: {', '.join(sorted(names))}\n")


def print_classified_symbols(symbols: List[ClassifiedSymbol]) -> None:
    d: Dict[Classification, List[str]] = defaultdict(list)
    for symbol in symbols:
        d[symbol.classification].append(symbol.symbol_name)
    print_classifications(d)


def print_all_symbols(symbols: SymbolCollection) -> None:
    print("FUNCTIONS")
    print_classified_symbols([function.name for function in symbols.functions])
    print("STRUCTS")
    print_classified_symbols([struct.name for struct in symbols.structs])
    print("GLOBAL VARS")
    print_classified_symbols([global_var.name for global_var in symbols.global_vars])

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


def print_statistics_and_get_score(statistics: Statistics) -> float:
    score = 0.0
    print("Total counts:")
    for symbol_type, counts in (
        ("functions", statistics.function_counts),
        ("structs", statistics.struct_counts),
        ("global vars", statistics.global_var_counts),
        ("struct members", statistics.struct_member_counts),
        ("local vars", statistics.local_var_counts),
    ):
        good = counts[Classification.GOOD]
        malformed = counts[Classification.MALFORMED]
        undocumented = counts[Classification.UNDOCUMENTED]
        total = good + malformed + undocumented
        print(f"{symbol_type}: {good}/{total} ({good / total * 100:.4f}%)")
        score += 0.2 * (good / total)
    return score


def get_git_rev(sm64_source: Path, commit_num: int) -> str:
    return (
        subprocess.run(
            [
                "/usr/bin/git",
                "-C",
                str(sm64_source),
                "rev-parse",
                f"master~{commit_num}",
            ],
            capture_output=True,
        )
        .stdout.decode("utf-8")
        .rstrip()
    )


def git_checkout(sm64_source: Path, commit_num: int, rev: str) -> None:
    print(f"...checking out master~{commit_num}...")
    subprocess.run(["/usr/bin/git", "-C", str(sm64_source), "checkout", rev])


def parse_cache(path: Path) -> SymbolCollection:
    with path.open(mode="rb") as open_file:
        return pickle.load(open_file)


def save_to_cache(path: Path, symbols: SymbolCollection):
    with path.open(mode="wb") as open_file:
        pickle.dump(symbols, open_file)


def print_everything(symbols: SymbolCollection) -> float:
    statistics = build_statistics(symbols)
    # print_all_symbols(symbols)
    score = print_statistics_and_get_score(statistics)
    print(f"final score: {score * 100:.4f}%")
    return score


def get_file_hash(path: Path):
    hasher = hashlib.new("md5")
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def collect_all_symbols(root: Path, overwrite_file_cache: bool) -> SymbolCollection:
    all_symbols = SymbolCollection()
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

        cache_file = Path(__file__).parent / "doc_cache" / get_file_hash(filename)
        if cache_file.is_file() and not overwrite_file_cache:
            file_symbols: SymbolCollection = parse_cache(cache_file)
        else:
            file_symbols = SymbolCollection()
            ast = get_ast_from_file(
                filename,
                [include_dir, src_dir, build_include_dir_jp, build_include_dir_us],
                [force_include_ultra64, force_include_sm64],
            )
            visitor = FileVisitor(str(filename))
            visitor.visit(ast)
            file_symbols.functions = visitor.functions
            file_symbols.structs = visitor.structs
            file_symbols.global_vars = collect_global_vars(ast)
            save_to_cache(cache_file, file_symbols)

        all_symbols.functions |= file_symbols.functions
        all_symbols.structs |= file_symbols.structs
        all_symbols.global_vars |= file_symbols.global_vars

    return all_symbols


@dataclass(frozen=True)
class CommitInfo:
    commit_hash: str
    timestamp: int
    author: str
    score: float
    num_coins: int


def analyze_commit(
    root: Path,
    overwrite_commit_cache: bool,
    overwrite_file_cache: bool,
    commit_num: int,
) -> CommitInfo:
    rev = get_git_rev(root, commit_num)
    cache = Path(__file__).parent / "doc_cache"
    cache.mkdir(exist_ok=True)
    cache_file = cache / rev
    if cache_file.is_file() and not overwrite_commit_cache:
        all_symbols = parse_cache(cache_file)
    else:
        git_checkout(root, commit_num, rev)
        all_symbols = collect_all_symbols(root, overwrite_file_cache)
        save_to_cache(cache_file, all_symbols)

    score = print_everything(all_symbols)
    return CommitInfo(
        commit_hash=rev,
        timestamp=git_get_timestamp(root, commit_num),
        author=git_get_author(root, commit_num),
        score=score,
        num_coins=score_to_coins(score),
    )


def git_format(sm64_source: Path, commit_num: int, format_str: str) -> str:
    return (
        subprocess.run(
            [
                "/usr/bin/git",
                "-C",
                str(sm64_source),
                "show",
                "-s",
                f"--format={format_str}",
                f"master~{commit_num}",
            ],
            capture_output=True,
        )
        .stdout.decode("utf-8")
        .rstrip()
    )


def git_get_author(sm64_source: Path, commit_num: int) -> str:
    return git_format(sm64_source, commit_num, "%an")


def git_get_timestamp(sm64_source: Path, commit_num: int) -> int:
    return (
        int(git_format(sm64_source, commit_num, "%ct"))
        # Flot wants milliseconds since epoch for some reason...
        * 1000
    )


def score_to_coins(percent: float) -> int:
    return round(percent * 2672)


def get_coin_leaderboard(commits: List[CommitInfo]) -> Dict[str, int]:
    d: Dict[str, int] = defaultdict(int)
    for i, commit in enumerate(commits):
        try:
            d[commit.author] += commit.num_coins - commits[i + 1].num_coins
        except IndexError:
            continue
    return d


def sm64_parse(
    root: Path,
    overwrite_commit_cache: bool = False,
    overwrite_file_cache: bool = False,
    commits_to_analyze: int = 1,
) -> List[List[Union[int, float]]]:
    # "results" is this stupid type to make converting to a Flot dataset
    # as easy as possible.
    commits: List[CommitInfo] = []
    results: List[List[Union[int, float]]] = []
    for commit_num in range(commits_to_analyze):
        print(f"analyzing commit master~{commit_num}...")
        commit_info: CommitInfo = analyze_commit(
            root, overwrite_commit_cache, overwrite_file_cache, commit_num
        )
        results.append([commit_info.timestamp, commit_info.num_coins])
        commits.append(commit_info)

    print("final results:")
    print(results)

    print("score leaderboard:")
    print(sorted(get_coin_leaderboard(commits).items(), key=lambda item: item[1]))
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", metavar="PATH_TO_SM64_SOURCE_DIR")
    parser.add_argument("--overwrite-commit-cache", action="store_true")
    parser.add_argument("--overwrite-file-cache", action="store_true")
    parser.add_argument("--commits-to-analyze", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.root)
    overwrite_commit_cache = args.overwrite_commit_cache
    overwrite_file_cache = args.overwrite_file_cache
    commits_to_analyze = args.commits_to_analyze
    sm64_parse(root, overwrite_commit_cache, overwrite_file_cache, commits_to_analyze)
    return 0


if __name__ == "__main__":
    sys.exit(main())
