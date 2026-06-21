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


def test_convention_beh_prefix_is_typo_for_bhv():
    # `beh_` shipped in the decomp (file_select.c menu backgrounds) as a
    # misspelling of the behavior native-entry prefix `bhv_`.
    assert (
        m.convention_violation(mksym("beh_yellow_background_menu_loop", "function"))
        == "behavior prefix typo: beh_ should be bhv_"
    )
    # FP-safety: only the `beh_` prefix is a typo, not any word starting "beh".
    assert m.convention_violation(mksym("set_mario_action", "function")) is None
    assert m.convention_violation(mksym("behind_camera_check", "function")) is None


def test_include_guard_convention():
    assert m.convention_violation(mksym("AREA_H", "guard")) is None
    assert m.convention_violation(mksym("WF_HEADER_H", "guard")) is None
    assert (
        m.convention_violation(mksym("MARIO_ACTIONS_MOVING", "guard"))
        == "include guard should end in _H (house style)"
    )
    assert (
        m.convention_violation(mksym(m._NO_GUARD, "guard"))
        == "header is missing an #ifndef include guard"
    )


def test_collect_guards(tmp_path):
    (tmp_path / "include").mkdir()
    (tmp_path / "src" / "game").mkdir(parents=True)
    (tmp_path / "levels").mkdir()
    (tmp_path / "include" / "area.h").write_text(
        "#ifndef AREA_H\n#define AREA_H\n#endif\n"
    )
    # a real decomp typo: guard missing the _H suffix
    (tmp_path / "src" / "game" / "moving.h").write_text(
        "#ifndef MARIO_ACTIONS_MOVING\n#define MARIO_ACTIONS_MOVING\n#endif\n"
    )
    # exempt X-macro data file: guardless on purpose, must be skipped (not flagged)
    (tmp_path / "levels" / "level_defines.h").write_text("DEFINE_LEVEL(...)\n")
    by_file = {g.file: g for g in m.collect_guards(tmp_path)}
    assert "levels/level_defines.h" not in by_file
    assert m.convention_violation(by_file["include/area.h"]) is None
    bad = by_file["src/game/moving.h"]
    assert bad.kind == "guard" and bad.name == "MARIO_ACTIONS_MOVING" and bad.line == 1
    assert m.convention_violation(bad)


def test_malformed_guard_not_double_counted_as_constant():
    # A guard missing _H must be measured as a guard, not slip through the
    # define loop as a conforming UPPER_SNAKE constant.
    src = (
        b"#ifndef MARIO_ACTIONS_MOVING\n#define MARIO_ACTIONS_MOVING\n"
        b"#define REAL_CONST 1\n#endif\n"
    )
    names = {(x.kind, x.name) for x in m.extract_source(src, "src/game/moving.h")}
    assert ("constant", "MARIO_ACTIONS_MOVING") not in names
    assert ("constant", "REAL_CONST") in names


def test_guard_rides_uniformity_but_not_completeness():
    # guard is a CONVENTION_KIND (scored on uniformity) but not a CLASSIFIER kind
    # (excluded from completeness counts, which iterate CLASSIFIERS).
    assert "guard" in m.CONVENTION_KINDS and "guard" not in m.CLASSIFIERS
    bad = mksym("BADGUARD", "guard")  # classification GOOD, but off-convention
    counts = m.category_counts([bad])  # must not KeyError on the unknown kind
    assert all(sum(v.values()) == 0 for v in counts.values())
    ucounts = m.uniformity_counts([bad])
    assert ucounts["guard"]["VIOLATION"] == 1


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


CONST_SOURCE = b"""
#ifndef _TEST_H_
#define _TEST_H_
#define ACT_WALKING 0x04
#define ACT_UNKNOWN_5 0x05
#define SURFACE_DEFAULT 0x00
#define MIN(a, b) ((a) < (b) ? (a) : (b))
enum Foo { FOO_A, FOO_B, special_lowercase, macro_unknown_28 };
#endif
"""


def test_defines_and_enums_extracted():
    s = m.extract_source(CONST_SOURCE, "x.h")
    constants = {x.name for x in s if x.kind == "constant"}
    enums = {x.name for x in s if x.kind == "enum"}
    assert {"ACT_WALKING", "ACT_UNKNOWN_5", "SURFACE_DEFAULT"} <= constants
    assert "_TEST_H_" not in constants  # include guard skipped
    assert "MIN" not in constants  # function-like macro is not a constant
    assert {"FOO_A", "FOO_B", "special_lowercase", "macro_unknown_28"} <= enums


def test_constant_placeholder_classification():
    assert m.classify_constant("ACT_WALKING") == C.GOOD
    assert m.classify_constant("SURFACE_DEFAULT") == C.GOOD  # no hex/address FP
    assert m.classify_constant("ACT_UNKNOWN_5") == C.UNDOCUMENTED
    assert m.classify_constant("FOO_80339876") == C.UNDOCUMENTED  # trailing address
    assert m.classify_constant("D_8033") == C.UNDOCUMENTED


def test_unused_is_documented_not_a_defect():
    # "unused" records real knowledge (matching decomp must keep unused content);
    # only the address-named ones are still incomplete. (UNK = ignorance, stays.)
    assert m.classify_constant("MODEL_DOOR_UNUSED") == C.GOOD
    assert m.classify_constant("SOUND_MENU_UNUSED") == C.GOOD
    assert m.classify_constant("UNUSED_COUNT_80333EE8") == C.UNDOCUMENTED  # address
    assert m.classify_constant("ACT_UNKNOWN_5") == C.UNDOCUMENTED  # unknown != unused
    assert m.classify_global_var("unused_ice_shard") == C.GOOD
    assert m.classify_global_var("gUnusedLoadedPool") == C.GOOD
    assert m.classify_global_var("sUnused80226B40") == C.UNDOCUMENTED  # address


def test_constant_upper_snake_convention():
    assert m.convention_violation(mksym("ACT_WALKING", "constant")) is None
    assert m.convention_violation(mksym("SOUND_GENERAL_COIN", "enum")) is None
    assert m.convention_violation(mksym("macro_yellow_coin_1", "enum"))
    assert m.convention_violation(mksym("AddToGroup", "constant"))  # goddard PascalCase


FAMILY_SOURCE = b"""
#define ACT_WALKING 0x04
#define ACT_JUMPING 0x05
#define ACT_RUNNING 0x06
#define SOLO_ONE_OFF 0x01
enum E { ACT_FLAG_AIR, ACT_FLAG_MOVING, ACT_FLAG_STATIONARY };
const BehaviorScript bhvGoomba[] = {};
"""


def test_family_of():
    g = {s.name: s for s in m.extract_source(FAMILY_SOURCE, "f.c")}
    assert m.family_of(g["ACT_WALKING"]) == "ACT_*"
    assert m.family_of(g["bhvGoomba"]) == "bhv*"
    # a function has no family
    fn = m.extract_source(b"void set_x(void) {}", "x.c")[0]
    assert m.family_of(fn) is None


def test_family_counts_collapses_small_groups():
    s = m.extract_source(FAMILY_SOURCE, "f.c")
    # ACT_ has 6 members (3 #define + 3 enum) -> named at threshold 4; SOLO -> other
    fams = m.family_counts(s, min_members=4)
    assert "ACT_*" in fams
    assert fams["ACT_*"]["count"] == 6
    assert "SOLO_*" not in fams
    assert m.OTHER_FAMILY in fams  # SOLO_ONE_OFF collapsed here


def test_to_json_has_families_and_row_family():
    d = m.to_json(m.extract_source(FAMILY_SOURCE, "f.c"))
    assert "families" in d
    # every needs_attention / violation row carries a family key (may be null)
    for row in d["needs_attention"] + d["violations"]:
        assert "family" in row


def test_to_json_has_semantic_and_no_family_uniformity():
    d = m.to_json(m.extract_source(FAMILY_SOURCE, "f.c"))
    assert "semantic_entities" in d and "semantic_findings" in d
    # per-family uniformity was tautological and is gone.
    if d["families"]:
        assert "conforming" not in next(iter(d["families"].values()))


def test_object_fields_are_their_own_kind_not_upper_snake_violations():
    # #define oFoo OBJECT_FIELD_*(...) is intentionally o+PascalCase. It must be a
    # distinct kind so it isn't scored as a UPPER_SNAKE constant violation.
    src = (
        b"#define oAction OBJECT_FIELD_S32(0x14)\n"
        b"#define oPosX OBJECT_FIELD_F32(0x06)\n"
        b"#define oUnk94 OBJECT_FIELD_S16(0x94)\n"
        b"#define ACT_WALKING 0x04\n"
    )
    s = m.extract_source(src, "include/object_fields.h")
    kinds = {x.name: x.kind for x in s}
    assert kinds["oAction"] == "object_field" and kinds["oPosX"] == "object_field"
    assert kinds["ACT_WALKING"] == "constant"  # real constants stay constants
    # o+PascalCase conforms; UPPER_SNAKE rule does NOT apply to object fields.
    assert m.convention_violation(find(s, "object_field", "oAction")) is None
    assert m.convention_violation(mksym("o_action", "object_field"))  # snake is wrong
    # oUnk94 is a placeholder on the completeness axis.
    assert find(s, "object_field", "oUnk94").classification == C.UNDOCUMENTED


def test_inner_union_members_are_counted():
    # Anonymous unions/structs expose their fields directly on the parent; those
    # members were previously dropped entirely.
    src = b"""
    struct Object {
        s16 visible;
        union {
            s32 asS32;
            struct { s16 x; s16 y; } asPoint;
        };
    };
    """
    members = {x.name for x in m.extract_source(src, "x.h") if x.kind == "member"}
    assert {"visible", "asS32", "asPoint"} <= members


def test_doc_comment_coverage(tmp_path):
    (tmp_path / "src" / "game").mkdir(parents=True)
    (tmp_path / "src" / "game" / "a.c").write_text(
        "/** does a thing */\nvoid documented_fn(void) {}\n"
        "void undocumented_fn(void) {}\n"
        "// not a doc comment\nvoid also_undocumented(void) {}\n"
    )
    doc = m.doc_comment_coverage(tmp_path)
    assert doc["total"]["functions"] == 3 and doc["total"]["documented"] == 1
    assert doc["by_area"]["game"] == {"functions": 3, "documented": 1}
    # surfaced in json when a root is given, omitted otherwise
    assert m.to_json([], tmp_path)["doc_comments"]["total"]["functions"] == 3
    assert m.to_json([])["doc_comments"] is None


def test_typedef_struct_members_extracted():
    # The bulk of SM64's types are anonymous `typedef struct { ... } Name;`. They
    # were previously skipped (no struct tag), under-counting structs and members.
    src = b"typedef struct { s16 health; u8 unk02; } MarioBodyState;\n"
    s = m.extract_source(src, "x.h")
    assert find(s, "struct", "MarioBodyState").classification == C.GOOD
    assert find(s, "member", "health").classification == C.GOOD
    assert find(s, "member", "unk02").classification == C.UNDOCUMENTED
    # A named typedef must not be double-counted as a struct.
    s2 = m.extract_source(b"typedef struct Foo { s32 x; } Foo;\n", "x.h")
    assert len([x for x in s2 if x.kind == "struct" and x.name == "Foo"]) == 1


def test_completeness_reason_is_specific_and_actionable():
    s = syms()
    # Every flagged symbol gets a non-empty, specific reason.
    for x in s:
        if x.classification != C.GOOD:
            assert m.completeness_reason(x), x.name
        else:
            assert m.completeness_reason(x) == ""
    # Spot-check that the reason names the actual pattern, not just "undocumented".
    assert "func_" in m.completeness_reason(find(s, "function", "func_80246EE0"))
    assert "D_" in m.completeness_reason(find(s, "global", "D_80339876"))
    assert "unk" in m.completeness_reason(find(s, "member", "unk02"))
    assert "stack slot" in m.completeness_reason(find(s, "local", "sp1C"))
    assert "arg" in m.completeness_reason(find(s, "arg", "arg0"))
    assert "filler" in m.completeness_reason(find(s, "member", "filler"))
    assert "uppercase" in m.completeness_reason(find(s, "global", "UnusedThing"))


def test_to_json_needs_attention_has_reason():
    rows = m.to_json(syms())["needs_attention"]
    assert rows and all(r["reason"] for r in rows)


# --- semantic entities that read extra repo files (Dialog / Level / Sequence) --- #


def _summary_for(symbols, root, entity):
    summaries, findings = m.semantic_report(symbols, root)
    summary = next((x for x in summaries if x["entity"] == entity), None)
    gaps = {f.name for f in findings if f.entity == entity}
    return summary, gaps


def test_semantic_dialog_crossref(tmp_path):
    (tmp_path / "text" / "us").mkdir(parents=True)
    (tmp_path / "text" / "us" / "dialogs.h").write_text(
        'DEFINE_DIALOG(DIALOG_000, 1, 6, 30, 200, _("hi"))\n'
    )
    src = b"enum DialogId { DIALOG_000, DIALOG_001, DIALOG_NONE };\n"
    symbols = m.extract_source(src, "include/dialog_ids.h")
    summary, gaps = _summary_for(symbols, tmp_path, "Dialog")
    assert summary["members"] == 2 and summary["linked"] == 1  # NONE excluded
    assert gaps == {"DIALOG_001"}


def test_semantic_level_crossref(tmp_path):
    (tmp_path / "levels" / "bbh").mkdir(parents=True)
    (tmp_path / "levels" / "bbh" / "script.c").write_text(
        "const LevelScript level_bbh_entry[] = {};\n"
    )
    (tmp_path / "levels").mkdir(exist_ok=True)
    (tmp_path / "levels" / "level_defines.h").write_text(
        'DEFINE_LEVEL("BBH", LEVEL_BBH, COURSE_BBH, bbh, spooky, 1, 0, 0, 0, _, _)\n'
        'DEFINE_LEVEL("GONE", LEVEL_GONE, COURSE_GONE, gone, x, 1, 0, 0, 0, _, _)\n'
    )
    summary, gaps = _summary_for([], tmp_path, "Level")
    assert summary["members"] == 2 and summary["linked"] == 1
    assert gaps == {"LEVEL_GONE"}


def test_semantic_sequence_crossref(tmp_path):
    (tmp_path / "sound").mkdir()
    (tmp_path / "sound" / "sequences.json").write_text(
        '{"00_sound_player": [], "03_level_grass": [], "01_cutscene_collect_star": []}'
    )
    src = (
        b"enum SeqId {\n"
        b"  SEQ_SOUND_PLAYER,\n"
        b"  SEQ_LEVEL_GRASS,\n"
        b"  SEQ_EVENT_CUTSCENE_COLLECT_STAR,\n"  # EVENT_ infix dropped in manifest
        b"  SEQ_MISSING,\n"
        b"  SEQ_COUNT\n"
        b"};\n"
    )
    symbols = m.extract_source(src, "include/seq_ids.h")
    summary, gaps = _summary_for(symbols, tmp_path, "Music sequence")
    assert summary["members"] == 4  # SEQ_COUNT excluded
    assert gaps == {"SEQ_MISSING"}


def test_file_based_entities_skipped_without_root():
    # Every semantic entity now reads repo files (only links the decomp is required
    # to have); with no root, none can run, so the section is empty rather than noisy.
    symbols = m.extract_source(b"#define ACT_WALKING 1\n", "include/sm64.h")
    summaries, findings = m.semantic_report(symbols, root=None)
    assert summaries == [] and findings == []


def test_asset_bucket_maps_c_type():
    assert m.asset_bucket("foo_dl", "const Gfx") == "display list"
    assert m.asset_bucket("foo_geo", "const GeoLayout") == "geo layout"
    assert m.asset_bucket("foo_05000048", "const Vtx") == "vertex"
    assert m.asset_bucket("foo_collision", "Collision") == "collision"
    assert m.asset_bucket("foo_lights", "const Lights1") == "lighting"
    assert m.asset_bucket("foo_anim", "const struct Animation") == "animation"
    # animation s16/u16 tables are caught by the name (type is too generic)
    assert (
        m.asset_bucket("birds_seg5_animindex_5000870", "static const s16")
        == "animation"
    )
    assert m.asset_bucket("foo_tex", "ALIGNED8 const Texture") == "texture"
    assert m.asset_bucket("bbh_seg7_macro_objs", "const MacroObject") == "level data"


def test_collect_assets(tmp_path):
    (tmp_path / "actors" / "bird").mkdir(parents=True)
    (tmp_path / "actors" / "bird" / "model.inc.c").write_text(
        "const Vtx birds_seg5_vertex_05000048[] = {};\n"  # address placeholder
        "const Gfx bird_wing_dl[] = {};\n"  # real name
        "const struct Animation birds_seg5_anim_050008D0 = {};\n"
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "behavior_data.c").write_text(
        "const BehaviorScript bhvBird[] = {};\n"
    )
    by_name = {a.name: a for a in m.collect_assets(tmp_path)}
    assert "bhvBird" not in by_name  # behaviors excluded (scored as code globals)
    vtx = by_name["birds_seg5_vertex_05000048"]
    assert vtx.kind == "vertex" and vtx.classification == C.UNDOCUMENTED  # ROM address
    assert by_name["bird_wing_dl"].kind == "display list"
    assert by_name["bird_wing_dl"].classification == C.GOOD
    assert by_name["birds_seg5_anim_050008D0"].kind == "animation"


def test_asset_report_per_type_lowest_first():
    assets = [
        m.Symbol("a_dl", "display list", C.GOOD, "x.c", 1),
        m.Symbol("b_05000048", "display list", C.UNDOCUMENTED, "x.c", 2),
        m.Symbol("c_05000048", "vertex", C.UNDOCUMENTED, "x.c", 3),
    ]
    rep = m.asset_report(assets)
    assert rep["total"] == {"named": 1, "total": 3}
    assert rep["by_type"]["display list"] == {"named": 1, "total": 2}
    assert rep["by_type"]["vertex"] == {"named": 0, "total": 1}
    # ordered lowest fraction first: vertex (0%) before display list (50%)
    assert list(rep["by_type"]) == ["vertex", "display list"]


def test_assets_do_not_pollute_code_scores():
    code = syms()
    base = m.to_json(code)
    assert base["assets"] is None and base["asset_findings"] == []
    assets = [
        m.Symbol("v_05000048", "vertex", C.UNDOCUMENTED, "actors/x.c", 1),
        m.Symbol("good_dl", "display list", C.GOOD, "actors/x.c", 2),
    ]
    d = m.to_json(code, assets=assets)
    assert d["assets"]["total"] == {"named": 1, "total": 2}
    assert len(d["asset_findings"]) == 1 and d["asset_findings"][0]["kind"] == "vertex"
    # the asset axis is fully separate: code completeness/worklist are untouched.
    assert d["score"] == base["score"]
    assert d["categories"] == base["categories"]
    assert len(d["needs_attention"]) == len(base["needs_attention"])


def test_html_has_sidebar_and_asset_section():
    data = m.to_json(
        syms(),
        assets=[m.Symbol("v_05000048", "vertex", C.UNDOCUMENTED, "actors/x.c", 1)],
    )
    html = report_html.render(data)
    assert html.count("</script>") == 1
    assert 'class="sidebar"' in html and html.count('class="nav-item"') == 7
    assert 'id="panel-assets"' in html and 'id="card-assets"' in html
    assert "asset data named" in html


def test_html_worklist_has_group_by_file():
    html = report_html.render(m.to_json(syms()))
    assert 'id="groupfile"' in html and 'id="collapseall"' in html
    assert "tr.group" in html  # grouped-row styling + JS path are present


def test_html_has_theme_toggle_and_reason(tmp_path):
    data = m.to_json(syms())
    html = report_html.render(data, label="x")
    assert "__DATA__" not in html and "__LABEL__" not in html  # tokens substituted
    assert 'id="theme"' in html and "data-theme" in html
    assert "prefers-color-scheme" in html


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
