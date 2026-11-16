"""Law index: consolidation (2008 replaces 2007), retrieval, and the search-only topic lines."""
import json

import pytest

import law_index
from law_index import (BM25, add_noise, evaluate, format_articles, gold_ids, load_articles, load_topics,
                       retrieve, rrf, split_articles, tokens)

SUPERSEDED_BY_2008 = {2, 3, 4, 9, 13, 14, 15, 25, 26, 29, 36}


# ----------------------------------------------------------------- building blocks
def test_split_articles_reads_numbered_headings():
    text = "preamble\nArticle (1)\nFirst body.\nArticle (2)\nSecond body\nover two lines.\n"
    assert split_articles(text) == [(1, "First body."), (2, "Second body\nover two lines.")]


def test_gold_ids_parses_law_references():
    assert gold_ids("Art. 28 (sale ...); Art. 25(2)(d) (sale only at expiry)") == {"L26-28", "L26-25"}
    assert gold_ids("Art. 9 (as amended by Law 33/2008); Decree 43/2013") == {"L26-9", "D43-1"}
    assert gold_ids("") == set()


def test_tokens_drop_stopwords_and_keep_numbers():
    assert tokens("The Tenant shall give 90 days' notice") == ["tenant", "give", "90", "days", "notice"]


def test_bm25_ranks_the_matching_document_first():
    bm = BM25(["security deposit refund", "rent increase at renewal", "maintenance and repairs"])
    scores = bm.scores("who pays for repairs")
    assert scores.index(max(scores)) == 2
    assert bm.scores("zebra") == [0.0, 0.0, 0.0]


def test_rrf_rewards_documents_ranked_high_by_both_lists():
    fused = rrf([[0, 1, 2], [1, 0, 2]])
    assert fused[0] == pytest.approx(fused[1]) and fused[0] > fused[2]
    assert rrf([[5, 6], [5, 7]])[5] > rrf([[5, 6], [5, 7]])[6]


def test_add_noise_is_reproducible_and_changes_text():
    import random
    text = "The Landlord may terminate this contract at any time."
    a = add_noise(text, 0.2, random.Random(1))
    assert a == add_noise(text, 0.2, random.Random(1)) and a != text
    assert add_noise(text, 0.0, random.Random(1)) == text


# ----------------------------------------------------------------- consolidation: 2008 replaces 2007
def test_superseded_2007_articles_are_never_retrievable():
    everything = load_articles(include_superseded=True)
    current = load_articles()
    assert {a["article"] for a in everything if a["superseded"]} == SUPERSEDED_BY_2008
    assert not any(a["superseded"] for a in current)
    assert len({a["id"] for a in current}) == len(current)
    for n in SUPERSEDED_BY_2008:
        art = next(a for a in current if a["id"] == f"L26-{n}")
        assert "33/2008" in art["source"]


def test_article_25_is_the_2008_text_with_twelve_months_notice():
    """The benchmark's key finding depends on this: eviction at expiry needs 12 months' notice
    (2008), not 90 days (2007)."""
    everything = {a["id"]: a for a in load_articles(include_superseded=True)}
    assert "twelve (12) months" in " ".join(everything["L26-25"]["text"].split())
    assert "ninety" in everything["L26-25-2007"]["text"].lower()
    assert all(a["id"] != "L26-25-2007" for a in retrieve("eviction notice ninety days", 50))


# ----------------------------------------------------------------- topic lines
def test_topic_lines_point_at_real_current_articles():
    ids = {a["id"] for a in load_articles()}
    topics = load_topics()
    assert topics and set(topics) <= ids
    assert "_comment" in json.load(open(law_index.TOPICS, encoding="utf-8"))


def test_topic_lines_are_search_only_never_in_the_prompt():
    arts = retrieve("penalty if the tenant leaves early", 3, "bm25-topics")
    for with_ids in (False, True):
        shown = format_articles(arts, with_ids=with_ids)
        assert "Topics:" not in shown
        for topic in load_topics().values():
            assert topic not in shown
    assert all("topic" not in a for a in arts)


def test_format_articles_default_is_unchanged_and_ids_are_opt_in():
    arts = retrieve("security deposit refund", 1)
    a = arts[0]
    assert format_articles(arts) == f"[{a['source']} - Article ({a['article']})]\n{a['text']}"
    assert format_articles(arts, with_ids=True).startswith(f"[{a['id']} | ")


# ----------------------------------------------------------------- retrieval quality (regression guards)
def test_plain_bm25_is_unchanged():
    """`bm25` must keep giving the published numbers (RESULTS.md 4.2), so old runs stay reproducible."""
    r = evaluate("bm25", verbose=False)
    assert r["hit"] == {1: 21, 3: 33, 5: 40}
    assert r["by_cat"]["early_termination"] == (0, 10)


def test_bm25_topics_fixes_early_termination_without_hurting_any_category():
    before = evaluate("bm25", verbose=False)
    after = evaluate("bm25-topics", verbose=False)
    assert after["by_cat"]["early_termination"][0] >= 8
    assert after["hit"][3] >= 50
    for cat, (hits, _) in before["by_cat"].items():
        assert after["by_cat"][cat][0] >= hits, cat


def test_bm25_topics_also_helps_on_clauses_written_before_tuning():
    before = evaluate("bm25", clauses="eval/retrieval_heldout.csv", verbose=False)
    after = evaluate("bm25-topics", clauses="eval/retrieval_heldout.csv", verbose=False)
    assert after["hit"][3] > before["hit"][3]


def test_early_termination_clause_now_finds_article_7():
    got = [a["id"] for a in retrieve("Early termination by the Tenant shall result in a penalty of four (4) "
                                     "months' rent.", 3, "bm25-topics")]
    assert "L26-7" in got


def test_fusion_combines_bm25_topics_and_embeddings(monkeypatch):
    """No API: embeddings are faked so that the embedding ranking always prefers Art. 16."""
    arts = load_articles()
    target = next(i for i, a in enumerate(arts) if a["id"] == "L26-16")

    def fake_embed(texts, model="text-embedding-3-small"):
        if len(texts) == len(arts):                       # the documents
            return [[1.0, 0.0] if i == target else [0.0, 1.0] for i in range(len(arts))]
        return [[1.0, 0.0] for _ in texts]                # the query: identical to Art. 16's vector

    monkeypatch.setattr(law_index, "_embed", fake_embed)
    monkeypatch.setitem(law_index._cache, "doc_vecs_topics", fake_embed(["x"] * len(arts)))
    got = [a["id"] for a in retrieve("The security deposit is non-refundable.", 3, "fusion")]
    assert "L26-16" in got and "L26-20" in got            # one from each ranking


def test_unknown_retriever_is_rejected():
    with pytest.raises(ValueError):
        retrieve("anything", 3, "nope")
