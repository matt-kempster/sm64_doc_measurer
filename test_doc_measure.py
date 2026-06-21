"""Tests for the modernized tree-sitter engine (doc_measure.py)."""

import doc_measure as m
from doc_measure import Classification as C

# A snippet that exercises every symbol category, with a deliberate mix of
# good / malformed / undocumented names so the classifiers are checked too.
SOURCE = b"""
struct MarioState {
    s16 health;        /* good member */
    u8 unk02;          /* undocumented member */
    u8 filler[2];      /* malformed member */
};

struct Dummy123 {      /* bss-reorder padding: must be skipped entirely */
    s32 x;
};

s32 gGoodGlobal;       /* good global */
s32 D_80339876;        /* undocumented global */
s32 UnusedThing;       /* malformed global (leading uppercase) */

void func_80246EE0(s32 arg0) {   /* undocumented function + undocumented arg */
    s32 sp1C;          /* undocumented local */
    s32 count;         /* good local */
}

s32 update_mario(struct MarioState *m, f32 speedValue) {  /* good fn + good args */
    s32 i = 0;
    return i;
}
"""


def syms():
    return m.extract_source(SOURCE, "src/test.c")


def find(symbols, kind, name):
    return next(s for s in symbols if s.kind == kind and s.name == name)


def test_every_category_is_extracted():
    kinds = {s.kind for s in syms()}
    assert kinds == {"function", "arg", "local", "struct", "member", "global"}


def test_struct_members_are_found():
    # Regression guard: members are `field_identifier` nodes, not `identifier`.
    members = {s.name for s in syms() if s.kind == "member"}
    assert {"health", "unk02", "filler"} <= members


def test_dummy_struct_is_skipped():
    s = syms()
    assert not any(x.kind == "struct" and x.name.startswith("Dummy") for x in s)
    assert not any(x.kind == "member" and x.name == "x" for x in s)


def test_classifications():
    s = syms()
    assert find(s, "member", "health").classification == C.GOOD
    assert find(s, "member", "unk02").classification == C.UNDOCUMENTED
    assert find(s, "member", "filler").classification == C.MALFORMED
    assert find(s, "global", "gGoodGlobal").classification == C.GOOD
    assert find(s, "global", "D_80339876").classification == C.UNDOCUMENTED
    assert find(s, "global", "UnusedThing").classification == C.MALFORMED
    assert find(s, "function", "func_80246EE0").classification == C.UNDOCUMENTED
    assert find(s, "function", "update_mario").classification == C.GOOD
    assert find(s, "local", "sp1C").classification == C.UNDOCUMENTED
    assert find(s, "local", "count").classification == C.GOOD
    assert find(s, "arg", "arg0").classification == C.UNDOCUMENTED


def test_provenance_is_recorded():
    health = find(syms(), "member", "health")
    assert health.file == "src/test.c"
    assert health.line == 3  # 1-based; struct opens on line 2


def test_score_and_counts():
    s = syms()
    counts = m.category_counts(s)
    # 2 functions, exactly 1 documented (update_mario).
    assert counts["function"][C.GOOD] == 1
    assert sum(counts["function"].values()) == 2
    score = m.overall_score(counts)
    assert 0.0 < score < 1.0


def test_preproc_blanking_keeps_guarded_code():
    src = b"""
#ifdef VERSION_JP
void guarded_fn(s32 arg0) {
    s32 sp10;
}
#endif
"""
    fns = {s.name for s in m.extract_source(src, "x.c") if s.kind == "function"}
    assert "guarded_fn" in fns
