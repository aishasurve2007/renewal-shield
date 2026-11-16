"""
Renewal Shield — law index + retrieval (Phase 3, step 3b: RAG).

1. BUILD: splits the law PDFs into articles and CONSOLIDATES them:
   Law 33/2008 says Articles 2, 3, 4, 9, 13, 14, 15, 25, 26, 29, 36 of Law 26/2007
   "are hereby superseded" — so for those, only the 2008 text goes in the index.
   The superseded 2007 versions are kept in the file for reference but never retrieved.
   Decree 43/2013 (rent increase caps) is added as well.

2. RETRIEVE: finds the articles most relevant to a clause.
     bm25         – keyword ranking over the law text, no API, instant (default)
     openai       – embeddings (text-embedding-3-small), needs OPENAI_API_KEY
     bm25-topics  – bm25, but each article is also searchable by a plain-language topic line
                    (law/topics.json). Fixes vocabulary gaps: clauses say "penalty for leaving early",
                    Art. 7 says "may not be unilaterally terminated". No API.
     fusion       – bm25-topics and embeddings combined by rank (reciprocal rank fusion)
   The topic lines are search-only. The LLM is always given the verbatim article text.
   bm25 and openai are unchanged, so earlier results stay reproducible.

3. EVAL: every clause in the bank has a law_ref (e.g. "Art. 28 ...; Art. 25(2)(d)"),
   so we can measure whether retrieval finds the right article (hit@k).

Expected files in law/ (names just need to contain the law number and year):
  *26*2007*en*.pdf     Law 26/2007 (English)
  *33*2008*.pdf        Law 33/2008 amendment (English)
  *43*2013*.pdf/.txt   Decree 43/2013

Usage:
  python law_index.py --build
  python law_index.py --query "The Landlord may terminate at any time with 2 months notice"
  python law_index.py --eval
  python law_index.py --eval --retriever openai
  python law_index.py --eval --retriever bm25-topics
  python law_index.py --eval --retriever bm25-topics --clauses eval/retrieval_heldout.csv
  python law_index.py --eval --retriever bm25-topics --noise 0.03     # simulated OCR errors
"""

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
from collections import Counter

LAW_DIR = "law"
INDEX = os.path.join(LAW_DIR, "articles.json")
TOPICS = os.path.join(LAW_DIR, "topics.json")
METHODS = ["bm25", "openai", "bm25-topics", "fusion"]
FUSION_DEPTH = 10          # how many articles each retriever contributes to "fusion"

HEADER = re.compile(r"^\s*Arti[cd]l?e\s*\(?\s*(\d+)\s*\)?\s*$", re.M)   # "Article (25)" alone on a line
NOISE = [
    re.compile(r"Law No\.\s*\(\d+\)\s*of\s*\d{4}\s*(Regulating|Amending)", re.I),   # page footers
    re.compile(r"Page \d+ of \d+", re.I),
    re.compile(r"^\s*©"),
    re.compile(r"Every effort has been made|reference must be made to the original Arabic|"
               r"Arabic text will prevail|Supreme Legislation Committee", re.I),
]


# ----------------------------------------------------------------- reading
def read_text(path):
    if path.lower().endswith(".txt"):
        return open(path, encoding="utf-8").read()
    import pypdf
    return "\n".join((p.extract_text() or "") for p in pypdf.PdfReader(path).pages)


def clean(text):
    lines = [ln for ln in text.splitlines() if not any(p.search(ln) for p in NOISE)]
    return re.sub(r"[ \t]+", " ", "\n".join(lines))


def split_articles(text):
    """[(number, text), ...] in document order."""
    parts = HEADER.split(text)
    out = []
    for i in range(1, len(parts) - 1, 2):
        body = re.sub(r"\n{2,}", "\n", parts[i + 1]).strip()
        out.append((int(parts[i]), body))
    return out


def find(pattern):
    hits = [p for p in glob.glob(os.path.join(LAW_DIR, "*")) if re.search(pattern, os.path.basename(p), re.I)
            and p.lower().endswith((".pdf", ".txt")) and "articles" not in p]
    return sorted(hits, key=lambda p: (not p.lower().endswith(".pdf"), p))  # prefer PDF over txt


# ----------------------------------------------------------------- build
def build():
    f26 = [p for p in find(r"26.*2007") if not re.search(r"_ar\b|arabic|[؀-ۿ]", p, re.I)]
    f33 = find(r"33.*2008")
    f43 = find(r"43.*2013")
    if not f26 or not f33:
        raise SystemExit(f"Need Law 26/2007 (English) and Law 33/2008 PDFs in {LAW_DIR}/ — found: {f26} {f33}")
    print(f"Law 26/2007: {f26[0]}\nLaw 33/2008: {f33[0]}\nDecree 43/2013: {f43[0] if f43 else '(not found, skipped)'}")

    original = split_articles(clean(read_text(f26[0])))

    t33 = clean(read_text(f33[0]))
    m = re.search(r"Articles\s*(\(\s*\d+\s*\)[^.]{0,200}?)of the Original Law are\s+hereby\s+superseded",
                  t33, re.S | re.I)
    superseded = [int(n) for n in re.findall(r"\((\d+)\)", m.group(1))] if m else []
    if not superseded:
        raise SystemExit("Could not find the list of superseded articles in Law 33/2008.")
    after = t33[t33.lower().find("superseded by the following"):]
    replacements, seen = {}, set()
    for num, body in split_articles(after):
        if num in seen or num not in superseded:      # second "Article (2)" = the amending law's own article
            break
        seen.add(num)
        replacements[num] = body
    missing = set(superseded) - set(replacements)
    if missing:
        print(f"WARNING: no replacement text found for superseded articles {sorted(missing)}")

    articles = []
    for num, body in original:
        if num in replacements:
            articles.append({"id": f"L26-{num}", "article": num, "source": "Law 26/2007, as amended by Law 33/2008",
                             "text": replacements[num], "superseded": False})
            articles.append({"id": f"L26-{num}-2007", "article": num, "source": "Law 26/2007 (original wording)",
                             "text": body, "superseded": True})
        else:
            articles.append({"id": f"L26-{num}", "article": num, "source": "Law 26/2007",
                             "text": body, "superseded": False})
    if f43:
        for num, body in split_articles(clean(read_text(f43[0]))):
            articles.append({"id": f"D43-{num}", "article": num, "source": "Decree 43/2013",
                             "text": body, "superseded": False})

    os.makedirs(LAW_DIR, exist_ok=True)
    json.dump(articles, open(INDEX, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    live = [a for a in articles if not a["superseded"]]
    print(f"\n{len(live)} current articles indexed ({len(articles) - len(live)} superseded 2007 versions kept "
          f"for reference only) → {INDEX}")
    print(f"Superseded by Law 33/2008: {superseded}")
    for a in live:
        print(f"  {a['id']:8s} {a['source']:40s} {a['text'][:70]!r}")


def load_articles(include_superseded=False):
    if not os.path.exists(INDEX):
        raise SystemExit(f"{INDEX} not found — run  python law_index.py --build  first.")
    arts = json.load(open(INDEX, encoding="utf-8"))
    return arts if include_superseded else [a for a in arts if not a["superseded"]]


# ----------------------------------------------------------------- BM25
STOP = set("""a an and are as at be by for from has have if in into is it its of on or such that the their
this to was were will with which shall may any all other than not no be been being""".split())


def tokens(text):
    return [w for w in re.findall(r"[a-z]+|\d+", text.lower()) if w not in STOP and len(w) > 1]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tokens(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        df = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query):
        q = tokens(query)
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            out.append(s)
        return out


# ----------------------------------------------------------------- embeddings (optional)
EMB_CACHE = os.path.join(LAW_DIR, "embedding_cache.json")
_emb = None


def _embed(texts, model="text-embedding-3-small"):
    import urllib.request
    global _emb
    if _emb is None:
        _emb = json.load(open(EMB_CACHE)) if os.path.exists(EMB_CACHE) else {}
    keys = [hashlib.sha1(f"{model}|{t}".encode()).hexdigest() for t in texts]
    todo = [t for t, k in zip(texts, keys) if k not in _emb]
    if todo:
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({"model": model, "input": todo}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            vecs = [d["embedding"] for d in json.loads(r.read().decode())["data"]]
        for t, v in zip(todo, vecs):
            _emb[hashlib.sha1(f"{model}|{t}".encode()).hexdigest()] = v
        json.dump(_emb, open(EMB_CACHE, "w"))
    return [_emb[k] for k in keys]


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# ----------------------------------------------------------------- retrieval API
_cache = {}


def load_topics():
    """{article id: plain-language topic line}. Search-only: never shown to the LLM."""
    if not os.path.exists(TOPICS):
        return {}
    return {k: v for k, v in json.load(open(TOPICS, encoding="utf-8")).items() if not k.startswith("_")}


def _docs(arts, with_topics=False):
    topics = load_topics() if with_topics else {}
    return [f"{a['source']} Article {a['article']}. {a['text']}"
            + (f" Topics: {topics[a['id']]}." if a["id"] in topics else "") for a in arts]


def rrf(rankings, c=60):
    """Reciprocal rank fusion: combine several rankings (lists of doc indexes, best first) into
    one score per doc. Uses ranks only, so BM25 scores and cosine similarities (different scales)
    can be mixed safely."""
    score = {}
    for order in rankings:
        for rank, i in enumerate(order):
            score[i] = score.get(i, 0.0) + 1.0 / (c + rank + 1)
    return score


def _scores(text, arts, method):
    """One score per article (higher = more relevant) for bm25 / openai / bm25-topics."""
    with_topics = method == "bm25-topics" or method == "openai-topics"
    if method in ("bm25", "bm25-topics"):
        bm = _cache.get(method) or _cache.setdefault(method, BM25(_docs(arts, with_topics)))
        return bm.scores(text)
    if method in ("openai", "openai-topics"):
        key = "doc_vecs" if method == "openai" else "doc_vecs_topics"
        if key not in _cache:
            _cache[key] = _embed(_docs(arts, with_topics))
        q = _embed([text])[0]
        return [_cos(q, v) for v in _cache[key]]
    raise ValueError(method)


def retrieve(text, k=3, method="bm25"):
    """Top-k current articles for a clause: list of article dicts (with 'score')."""
    arts = _cache.get("arts") or _cache.setdefault("arts", load_articles())
    if method == "fusion":
        ranks = []
        for m in ("bm25-topics", "openai-topics"):
            sc = _scores(text, arts, m)
            # each retriever votes with its top FUSION_DEPTH only; articles sharing no word with the
            # clause (BM25 score 0) get no vote instead of an arbitrary rank
            ranks.append([i for i in sorted(range(len(arts)), key=lambda i: -sc[i]) if sc[i] > 0][:FUSION_DEPTH])
        fused = rrf(ranks)
        sc = [fused.get(i, 0.0) for i in range(len(arts))]
    elif method in ("bm25", "openai", "bm25-topics"):
        sc = _scores(text, arts, method)
    else:
        raise ValueError(method)
    order = sorted(range(len(arts)), key=lambda i: -sc[i])[:k]
    return [dict(arts[i], score=round(sc[i], 4)) for i in order]


def gold_ids(law_ref):
    """'Art. 28 (...); Art. 25(2)(d) ...; Decree 43/2013' -> {'L26-28', 'L26-25', 'D43-1'}"""
    ids = {f"L26-{n}" for n in re.findall(r"Art\.?\s*(\d+)", str(law_ref))}
    if re.search(r"Decree\s*43", str(law_ref), re.I):
        ids.add("D43-1")
    return ids


def format_articles(arts, with_ids=False):
    """Verbatim articles for the prompt. with_ids (prompt v4+) adds the article id to each heading
    so the model can say which article it relied on. The default output is unchanged from v1-v3."""
    if with_ids:
        return "\n\n".join(f"[{a['id']} | {a['source']} - Article ({a['article']})]\n{a['text']}" for a in arts)
    return "\n\n".join(f"[{a['source']} - Article ({a['article']})]\n{a['text']}" for a in arts)


# ----------------------------------------------------------------- eval
def add_noise(text, rate, rng):
    """Simulate OCR character errors: each letter/digit is replaced, dropped or doubled with
    probability `rate` (RESULTS.md reports ~2-3% character error on photos)."""
    out = []
    for ch in text:
        if ch.isalnum() and rng.random() < rate:
            op = rng.choice("rdx")
            if op == "r":
                out.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))
            elif op == "x":
                out.append(ch + ch)
        else:
            out.append(ch)
    return "".join(out)


def load_eval_clauses(path=None):
    """The clause bank, or any CSV with columns id, text, category, law_ref."""
    if path:
        import pandas as pd
        df = pd.read_csv(path).fillna("")
        print(f"Clauses: {path} ({len(df)} clauses)")
        return df
    from classify_baseline import load_bank
    return load_bank()


def evaluate(method, ks=(1, 3, 5), clauses=None, noise=0.0, seed=0, verbose=True):
    """hit@k for one retriever. Returns {'n', 'hit': {k: count}, 'by_cat': {category: (hits@3, n)}}."""
    df = load_eval_clauses(clauses)
    rng = random.Random(seed)
    res = {k: [] for k in ks}
    misses = []
    for _, r in df.iterrows():
        gold = gold_ids(r["law_ref"])
        text = add_noise(r["text"], noise, rng) if noise else r["text"]
        got = [a["id"] for a in retrieve(text, max(ks), method)]
        for k in ks:
            res[k].append(bool(gold & set(got[:k])))
        if not gold & set(got[:3]):
            misses.append((r["id"], sorted(gold), got[:3]))
    by_cat = {}
    for (_, r), hit in zip(df.iterrows(), res[3]):
        by_cat.setdefault(r["category"], []).append(hit)
    out = {"n": len(df), "hit": {k: sum(v) for k, v in res.items()},
           "by_cat": {c: (sum(v), len(v)) for c, v in by_cat.items()}}
    if verbose:
        tag = method + (f", {noise:.0%} simulated OCR noise, seed {seed}" if noise else "")
        print(f"\nRetrieval ({tag}) — does a correct article appear in the top k?")
        for k in ks:
            print(f"  hit@{k}: {sum(res[k])}/{len(res[k])} = {sum(res[k]) / len(res[k]):.1%}")
        print("  hit@3 by category: " + ", ".join(f"{c} {h}/{n}" for c, (h, n) in out["by_cat"].items()))
        if misses:
            print("\nMisses at k=3 (clause, correct, retrieved):")
            for m in misses:
                print(f"  {m[0]:7s} want {m[1]}  got {m[2]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query", default=None)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--retriever", choices=METHODS, default="bm25")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--clauses", default=None, help="--eval on another clause CSV (id,text,category,law_ref)")
    ap.add_argument("--noise", type=float, default=0.0, help="--eval with simulated OCR character errors, e.g. 0.03")
    ap.add_argument("--seed", type=int, default=0, help="random seed for --noise")
    args = ap.parse_args()
    if args.build:
        build()
    if args.query:
        for a in retrieve(args.query, args.k, args.retriever):
            print(f"{a['score']:8.3f}  {a['id']:8s} {a['source']}\n          {a['text'][:200]!r}\n")
    if args.eval:
        if args.retriever in ("openai", "fusion"):
            from classify_llm import load_dotenv
            load_dotenv()
        evaluate(args.retriever, clauses=args.clauses, noise=args.noise, seed=args.seed)


if __name__ == "__main__":
    main()
