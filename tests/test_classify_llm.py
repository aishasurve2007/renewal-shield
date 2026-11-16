"""LLM classifier: prompts per version, answer parsing, citations, cache. No API calls."""
import hashlib
import json

import pandas as pd
import pytest

import classify_llm
from classify_llm import (LAW_RULES, LAW_TEXT_INTRO, SYSTEM, build_prompt, cache_key, classify, metrics,
                          parse_answer, parse_article)
from law_index import retrieve

CLAUSE = "The Landlord may require the Tenant to vacate at the end of the contract by giving three (3) months' notice."


def arts():
    return retrieve(CLAUSE, 3, "bm25-topics")


# ----------------------------------------------------------------- prompt versions
def test_v3_prompt_text_is_frozen():
    """Benchmark rule: a prompt version is never edited in place. If this fails you changed v3:
    add a new version instead (see PROMPT_VERSION in classify_llm.py)."""
    digest = hashlib.sha1((SYSTEM + LAW_RULES + LAW_TEXT_INTRO).encode()).hexdigest()
    assert digest == "1383c598bbd48a280aa66a0c4b22ad49f6b307a9"


def test_default_version_is_v3_and_has_no_article_field():
    system, user = build_prompt(CLAUSE, "hybrid", arts())
    assert build_prompt(CLAUSE, "hybrid", arts(), "v3") == (system, user)
    assert '"article"' not in system and " | " not in system
    assert user == f'Clause: "{CLAUSE}"'


def test_v4_adds_only_the_article_field_and_article_ids():
    a = arts()
    v3, _ = build_prompt(CLAUSE, "hybrid", a, "v3")
    v4, _ = build_prompt(CLAUSE, "hybrid", a, "v4")
    assert '"article"' in v4
    for art in a:
        assert f"[{art['id']} | " in v4 and art["text"] in v4 and art["text"] in v3
    assert LAW_RULES in v4 and LAW_TEXT_INTRO in v4


def test_v4_equals_v3_when_no_articles_are_shown():
    for mode in ("zero", "law"):
        assert build_prompt(CLAUSE, mode, None, "v4") == build_prompt(CLAUSE, mode, None, "v3")


def test_unknown_version_is_rejected():
    with pytest.raises(ValueError):
        build_prompt(CLAUSE, "zero", None, "v9")


def test_cache_keys_differ_by_version_mode_and_articles():
    a = arts()
    keys = {cache_key("openai", "m", "hybrid", CLAUSE, a, "v3"), cache_key("openai", "m", "hybrid", CLAUSE, a, "v4"),
            cache_key("openai", "m", "rag", CLAUSE, a, "v3"), cache_key("openai", "m", "hybrid", CLAUSE, a[:2], "v3")}
    assert len(keys) == 4
    assert cache_key("openai", "m", "hybrid", CLAUSE, a) == cache_key("openai", "m", "hybrid", CLAUSE, a, "v3")


# ----------------------------------------------------------------- answer parsing
def test_parse_clean_answer():
    out = parse_answer('{"category": "deposit", "label": "problem", "severity": "red", "reason": "x"}')
    assert (out["category"], out["label"], out["severity"], out["repaired"]) == ("deposit", "problem", "red", False)
    assert "article" not in out


def test_parse_answer_wrapped_in_text_or_code_fence():
    raw = 'Sure:\n```json\n{"category": "Maintenance", "label": "FINE", "severity": "none", "reason": "ok"}\n```'
    out = parse_answer(raw)
    assert (out["category"], out["label"], out["severity"]) == ("maintenance", "fine", "none")


@pytest.mark.parametrize("label,severity,want", [
    ("red", "", ("problem", "red")),            # severity written in the label field
    ("yellow", "none", ("problem", "yellow")),
    ("ok", "none", ("fine", "none")),
    ("", "yellow", ("problem", "yellow")),      # label missing, severity says problem
    ("", "none", ("fine", "none")),
])
def test_parse_repairs_off_format_answers(label, severity, want):
    out = parse_answer(json.dumps({"category": "other", "label": label, "severity": severity}))
    assert (out["label"], out["severity"]) == want and out["repaired"]


def test_fine_always_means_severity_none():
    out = parse_answer('{"category": "other", "label": "fine", "severity": "yellow"}')
    assert (out["label"], out["severity"]) == ("fine", "none")


@pytest.mark.parametrize("raw", ["", None, "not json", "{broken", '{"label": "maybe"}'])
def test_parse_garbage_is_invalid_not_a_crash(raw):
    out = parse_answer(raw)
    assert out["label"] == "invalid" and out["category"] == "invalid"


def test_unknown_category_is_invalid():
    assert parse_answer('{"category": "parking", "label": "fine", "severity": "none"}')["category"] == "invalid"


# ----------------------------------------------------------------- citations (v4)
def test_citation_must_be_an_article_that_was_shown():
    shown = ["L26-25", "L26-7", "D43-1"]
    assert parse_article("L26-25", shown) == "L26-25"
    assert parse_article(" l26-7 ", shown) == "L26-7"
    assert parse_article("[D43-1]", shown) == "D43-1"
    assert parse_article("none", shown) == "none" and parse_article(None, shown) == "none"
    assert parse_article("L26-99", shown) == "invalid"        # invented article: never accepted
    assert parse_article("L26-14", shown) == "invalid"        # real article, but not shown
    assert parse_article("Article 25", shown) == "invalid"


def test_parse_answer_checks_the_article_when_ids_are_given():
    raw = '{"category": "deposit", "label": "problem", "severity": "red", "reason": "x", "article": "L26-20"}'
    assert parse_answer(raw, ["L26-20", "L26-21"])["article"] == "L26-20"
    assert parse_answer(raw, ["L26-7"])["article"] == "invalid"
    assert parse_answer(raw.replace(', "article": "L26-20"', ""), ["L26-20"])["article"] == "none"


def test_citation_metrics():
    rows = pd.DataFrame([
        # right article shown and cited
        dict(pred_article="L26-25", articles="L26-25|L26-7", gold_articles="L26-25"),
        # right article shown, wrong one cited
        dict(pred_article="L26-7", articles="L26-25|L26-7", gold_articles="L26-25"),
        # right article not retrieved: cannot be cited
        dict(pred_article="none", articles="L26-5|L26-6", gold_articles="L26-7"),
        # invented id
        dict(pred_article="invalid", articles="L26-20|L26-21", gold_articles="L26-20|L26-21"),
        # no law reference ("other" clause): not scored
        dict(pred_article="none", articles="L26-5", gold_articles=""),
    ])
    for col, val in (("gold_category", "x"), ("pred_category", "x"), ("gold_label", "fine"), ("pred_label", "fine"),
                     ("gold_severity", "none"), ("pred_severity", "none"), ("difficulty", "easy"), ("repaired", False)):
        rows[col] = val
    m = metrics(rows)
    assert m["citation"] == pytest.approx(1 / 4)
    assert m["citation_when_shown"] == pytest.approx(1 / 3)
    assert m["citation_invalid"] == pytest.approx(1 / 4)


def test_v3_results_have_no_citation_metric():
    rows = pd.DataFrame([dict(articles="L26-25", gold_articles="L26-25", gold_category="x", pred_category="x",
                              gold_label="fine", pred_label="fine", gold_severity="none", pred_severity="none",
                              difficulty="easy", repaired=False)])
    m = metrics(rows)
    assert "citation" not in m and m["retrieval_hit"] == 1.0


# ----------------------------------------------------------------- cache
def test_classify_calls_the_model_once_then_uses_the_cache(monkeypatch, tmp_path):
    calls = []

    def fake_llm(provider, model, system, user):
        calls.append(system)
        return '{"category": "landlord_termination", "label": "problem", "severity": "red", "reason": "r", "article": "L26-25"}'

    monkeypatch.setattr(classify_llm, "call_llm", fake_llm)
    monkeypatch.setattr(classify_llm, "CACHE", str(tmp_path / "cache.jsonl"))
    a, cache = arts(), {}
    first = classify("openai", "m", "hybrid", CLAUSE, cache, 0, a, "v4")
    again = classify("openai", "m", "hybrid", CLAUSE, cache, 0, a, "v4")
    assert first == again and len(calls) == 1
    assert first["article"] == ("L26-25" if "L26-25" in [x["id"] for x in a] else "invalid")
    assert classify_llm.load_cache()[cache_key("openai", "m", "hybrid", CLAUSE, a, "v4")]["answer"] == first
    v3 = classify("openai", "m", "hybrid", CLAUSE, cache, 0, a, "v3")        # other version: new call, no article
    assert len(calls) == 2 and "article" not in v3


def test_committed_cache_is_readable_and_never_shrinks():
    """results/llm_cache.jsonl is the record that makes published numbers reproducible: append only."""
    cache = classify_llm.load_cache()
    assert len(cache) >= 600
