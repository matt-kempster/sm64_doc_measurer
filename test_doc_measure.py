"""Tests for the modernized tree-sitter engine (doc_measure.py)."""

import json
import re

import doc_measure as m
import report_html
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


def test_arg_real_words_not_flagged_as_stack_slots():
    # Regression: the original "starts with sp and short" rule flagged real
    # words. Only sp+hex offsets (and not space/speed) are stack slots.
    for good in ("speed", "space", "spawn", "split"):
        assert m.classify_arg(good) == C.GOOD, good
    for placeholder in ("sp1C", "sp24", "spC"):
        assert m.classify_arg(placeholder) == C.UNDOCUMENTED, placeholder


def test_allcaps_macro_not_counted_as_function():
    # BAD_RETURN(s32) and friends are function-like macros, not functions.
    src = b"s32 BAD_RETURN(s32) real_func(void) { s32 i; return i; }\n"
    fns = {s.name for s in m.extract_source(src, "x.c") if s.kind == "function"}
    assert "BAD_RETURN" not in fns


def test_html_report_embeds_parseable_json():
    data = m.to_json(syms())
    html = report_html.render(data, label="test build")
    assert "<!doctype html>" in html
    assert html.count("</script>") == 1  # embedded data didn't close it early
    assert "test build" in html
    blob = re.search(r"const DATA = (\{.*?\});", html, re.S).group(1)
    parsed = json.loads(blob)
    assert parsed["categories"].keys() == data["categories"].keys()
    assert len(parsed["needs_attention"]) == len(data["needs_attention"])


def test_html_report_escapes_script_close():
    # A symbol name containing "</script>" must not break out of the <script>.
    data = {
        "score": 1.0,
        "categories": {},
        "needs_attention": [
            {
                "name": "</script>",
                "kind": "global",
                "classification": "MALFORMED",
                "file": "x.c",
                "line": 1,
            }
        ],
    }
    html = report_html.render(data)
    assert html.count("</script>") == 1


def mksym(name, kind, type_name=None):
    return m.Symbol(name, kind, C.GOOD, "x.c", 1, type_name)


def test_convention_casing_rules():
    assert m.convention_violation(mksym("set_mario_action", "function")) is None
    assert m.convention_violation(mksym("setMarioAction", "function"))
    assert m.convention_violation(mksym("MarioState", "struct")) is None
    assert m.convention_violation(mksym("mario_state", "struct"))
    assert m.convention_violation(mksym("rawStickX", "member")) is None
    assert m.convention_violation(mksym("raw_stick_x", "member"))


def test_convention_global_prefix_rules():
    assert m.convention_violation(mksym("gMarioStates", "global")) is None
    assert m.convention_violation(mksym("sIntangibleTimer", "global")) is None
    # snake_case data tables and _linker symbols are accepted alternatives.
    assert m.convention_violation(mksym("dialog_table_eu_en", "global")) is None
    assert m.convention_violation(mksym("_engineSegmentStart", "global")) is None
    # camelCase global with no g/s prefix is a genuine inconsistency.
    assert m.convention_violation(mksym("audioString34", "global"))


def test_convention_behavior_type_rule():
    src = (
        b"const BehaviorScript bhvGoomba[] = {};\nconst BehaviorScript behBad[] = {};\n"
    )
    g = {s.name: s for s in m.extract_source(src, "b.c") if s.kind == "global"}
    assert "BehaviorScript" in (g["bhvGoomba"].type_name or "")
    assert m.convention_violation(g["bhvGoomba"]) is None
    assert m.convention_violation(g["behBad"]) == "behavior should be bhv + PascalCase"


def test_uniformity_axis_excludes_args_and_locals():
    counts = m.uniformity_counts(syms())
    assert set(counts) == set(m.CONVENTION_KINDS)
    assert "arg" not in counts and "local" not in counts
    # Only layer-1 GOOD names are scored; undocumented placeholders excluded.
    assert m.uniformity_score(counts) == 1.0  # snippet's good names all conform


def test_to_json_has_uniformity_block():
    d = m.to_json(syms())
    assert "uniformity_score" in d
    assert set(d["conventions"]) == set(m.CONVENTION_KINDS)
    assert isinstance(d["violations"], list)


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
