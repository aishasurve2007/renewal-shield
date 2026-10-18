"""
Renewal Shield — Phase 3, step 3: LLM clause classifier.

Same task and same output as the baseline (category / problem-or-fine / severity),
so the numbers compare directly. Two modes:

  zero  – the model only gets the category descriptions. Tests what it already knows.
  law   – the model also gets short rules from Law 26/2007 (as amended by 33/2008)
          and Decree 43/2013. Tests whether giving it the law helps.

No training happens, so no leave-one-out is needed: every clause is unseen.

Providers (no SDK needed — plain HTTP from the standard library):
  gemini     env GEMINI_API_KEY     default model gemini-3.8-flash   (free tier ~10 req/min)
  openai     env OPENAI_API_KEY     default model gpt-4.1-mini
  anthropic  env ANTHROPIC_API_KEY  default model claude-haiku-5-5
  ollama     local, no key          default model qwen2.5:7b         (run: ollama pull qwen2.5:7b)
Model names change often — override with --model if a default stops working.

API keys are read from a .env file in the project folder (see .env.example), or from the terminal.

Every answer is cached in results/llm_cache.jsonl, so re-runs are free and an
interrupted run continues where it stopped.

Usage:
  python classify_llm.py --provider gemini --mode zero --dry-run      # show the prompt, no API call
  python classify_llm.py --provider gemini --mode zero --limit 5      # quick test on 5 clauses
  python classify_llm.py --provider gemini --mode zero
  python classify_llm.py --provider gemini --mode law
  python classify_llm.py --provider gemini --mode law --ocr-dir out   # also test on OCR'd photo text
  python classify_llm.py --compare                                    # table of every run so far
"""

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

import pandas as pd

from classify_baseline import load_bank, load_ocr_variants

CATEGORIES = {
    "rent_increase": "how the rent can change when the contract is renewed",
    "notice_period": "how much notice the landlord must give before changing rent or terms at renewal",
    "landlord_termination": "when and how the landlord can end the contract or evict the tenant",
    "early_termination": "what happens if the tenant leaves before the end of the contract",
    "maintenance": "who is responsible for repairs and maintenance",
    "deposit": "the security deposit: refund, deductions, timing",
}

LAW_RULES = """Relevant Dubai law (Law 26/2007 as amended by Law 33/2008; Decree 43/2013):
- Rent increase (Art. 9, 13; Decree 43/2013): increases at renewal are capped by the RERA/DLD rental index.
  A clause fixing an increase in advance (a %, an amount, "market rates", landlord's discretion) or stopping the
  tenant from disputing it conflicts with the law -> red. A clause that follows the index or the law -> fine.
- Notice of changes (Art. 14): a party wanting to change rent or terms at renewal must notify the other at least
  90 days before expiry, unless the parties agree otherwise. An agreed shorter period is legal but bad for the
  tenant -> yellow. No real notice at all ("at any time", new terms applying automatically) -> red.
  90 days or more, or the tenant's own notice, -> fine.
- Landlord ending the contract (Art. 25, 28): during the term only for tenant breaches listed in the law
  (rent unpaid 30 days after notice, unauthorised subletting, illegal use, damage, breach after 30 days' notice).
  At expiry only for demolition/reconstruction, major renovation, personal use by the owner or first-degree
  relatives, or sale - with 12 months' notice through a notary public or registered mail. A sale does not end a
  fixed-term lease (Art. 28). Termination outside these grounds or with shorter/informal notice -> red.
- Tenant leaving early (Art. 7): no unilateral termination during the term; early exit is a matter of contract.
  A penalty of about 1-2 months' rent with notice is normal -> fine. Full remaining rent, uncapped losses,
  forfeiting all cheques, or no exit at all -> yellow (negotiate).
- Maintenance (Art. 16): the landlord is responsible unless agreed otherwise. Tenant handling minor repairs,
  consumables or damage they caused -> fine. All, major or structural repairs, unlimited cost, or an "as is"
  waiver -> yellow (negotiate).
- Deposit (Art. 20, 21): refunded on expiry minus legitimate costs; ordinary wear and tear is excepted.
  Non-refundable or forfeited in full -> red. Deductions at the landlord's sole discretion, no refund deadline, or
  fixed charges regardless of condition -> yellow. Proof-based deductions, unpaid bills, clear timelines -> fine."""

SYSTEM = """You review clauses from Dubai residential tenancy contracts for tenants.
For the clause given, answer with ONLY a JSON object:
{"category": one of %s,
 "label": "problem" or "fine",
 "severity": "red" (likely conflicts with the law), "yellow" (legal but unfavourable, worth negotiating)
             or "none" (only when label is "fine"),
 "reason": one short sentence}
Rule: "label" is "problem" whenever "severity" is "red" or "yellow"; "fine" only when "severity" is "none".

Categories:
%s""" % (list(CATEGORIES), "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items()))

def load_dotenv(path=".env"):
    """Read KEY=value lines from .env into the environment (keys already set in the terminal win)."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8-sig"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()

DEFAULTS = {
    "gemini": ("gemini-3.8-flash", "GEMINI_API_KEY", 6.5),
    "openai": ("gpt-4.1-mini", "OPENAI_API_KEY", 0.5),
    "anthropic": ("claude-haiku-5-5", "ANTHROPIC_API_KEY", 0.5),
    "ollama": ("qwen2.5:7b", None, 0.0),
}
CACHE = "results/llm_cache.jsonl"
# Bump when the prompt changes: answers are cached per prompt version, and result
# files are named by it, so every version's numbers stay comparable in --compare.
#   v1: first prompt
#   v2: explicit rule that yellow/red => problem (v1 sometimes said "unfavourable" but labelled "fine")
PROMPT_VERSION = "v2"


# ----------------------------------------------------------------- prompts
def build_prompt(text, mode):
    system = SYSTEM + ("\n\n" + LAW_RULES if mode == "law" else "")
    return system, f'Clause: "{text}"'


# ----------------------------------------------------------------- providers
def _post(url, payload, headers, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def call_llm(provider, model, system, user):
    if provider == "gemini":
        key = os.environ["GEMINI_API_KEY"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        out = _post(url, {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }, {})
        return out["candidates"][0]["content"]["parts"][0]["text"]
    if provider == "openai":
        out = _post("https://api.openai.com/v1/chat/completions", {
            "model": model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }, {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        return out["choices"][0]["message"]["content"]
    if provider == "anthropic":
        out = _post("https://api.anthropic.com/v1/messages", {
            "model": model, "max_tokens": 300, "temperature": 0, "system": system,
            "messages": [{"role": "user", "content": user}],
        }, {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
        return out["content"][0]["text"]
    if provider == "ollama":
        out = _post("http://localhost:11434/api/chat", {
            "model": model, "stream": False, "format": "json", "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }, {}, timeout=300)
        return out["message"]["content"]
    raise ValueError(provider)


def parse_answer(raw):
    """Pull the JSON object out of the reply and normalise it."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    cat = str(d.get("category", "")).strip().lower()
    label = str(d.get("label", "")).strip().lower()
    sev = str(d.get("severity", "")).strip().lower()
    repaired = False
    if label in ("red", "yellow"):                    # model put the severity in the label field
        label, sev, repaired = "problem", label, True
    elif label in ("none", "ok", "acceptable", "compliant", "legal"):
        label, repaired = "fine", True
    elif label not in ("problem", "fine"):
        if sev in ("red", "yellow"):                  # label missing/odd but severity says problem
            label, repaired = "problem", True
        elif sev == "none":
            label, repaired = "fine", True
        else:
            label = "invalid"
    if label == "fine":
        sev = "none"
    if sev not in ("red", "yellow", "none"):
        sev = "invalid"
    return {"category": cat if cat in CATEGORIES else "invalid",
            "label": label, "severity": sev, "repaired": repaired,
            "reason": str(d.get("reason", ""))[:300]}


# ----------------------------------------------------------------- cache
def load_cache():
    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE, encoding="utf-8"):
            try:
                r = json.loads(line)
                cache[r["key"]] = r
            except Exception:
                pass
    return cache


def cache_key(provider, model, mode, text):
    system, user = build_prompt(text, mode)
    return hashlib.sha1(f"{provider}|{model}|{mode}|{system}|{user}".encode()).hexdigest()


def classify(provider, model, mode, text, cache, delay):
    key = cache_key(provider, model, mode, text)
    if key in cache:
        return cache[key]["answer"]
    system, user = build_prompt(text, mode)
    for attempt in range(5):
        try:
            raw = call_llm(provider, model, system, user)
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")[:300]
            if e.code in (429, 500, 503) and attempt < 4:          # rate limit / busy: back off
                wait = 20 * (attempt + 1)
                print(f"   HTTP {e.code}, waiting {wait}s ...")
                time.sleep(wait)
                continue
            if e.code == 429:
                raise SystemExit("Quota used up (HTTP 429). Answers so far are saved — run the same command "
                                 "again later (Gemini's free daily quota resets every 24h) and it will continue.")
            raise SystemExit(f"API error {e.code}: {body}")
    ans = parse_answer(raw)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "provider": provider, "model": model, "mode": mode,
                            "text": text, "raw": raw, "answer": ans}, ensure_ascii=False) + "\n")
    cache[key] = {"answer": ans}
    time.sleep(delay)
    return ans


# ----------------------------------------------------------------- evaluation
def run(args):
    model, env, delay = DEFAULTS[args.provider]
    model = args.model or model
    delay = args.delay if args.delay is not None else delay
    if env and not os.environ.get(env) and not args.dry_run:
        raise SystemExit(f"{env} not found. Add a line  {env}=your-key  to the .env file in the project folder.")

    df = load_bank(args.bank)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=0)

    if args.dry_run:
        system, user = build_prompt(df.iloc[0]["text"], args.mode)
        print("=== SYSTEM ===\n" + system + "\n\n=== USER ===\n" + user)
        return

    tests = [(r, "clean_text", r["text"]) for _, r in df.iterrows()]
    if args.ocr_dir:
        variants = load_ocr_variants(args.ocr_dir)
        for _, r in df.iterrows():
            photos = [t for k, t in variants.get(r["id"], []) if k == "photo"]
            if photos:
                tests.append((r, "ocr_photo", photos[0]))   # one photo version per clause keeps API use low

    cache = load_cache()
    rows = []
    print(f"{args.provider} / {model} / mode={args.mode}: {len(tests)} clauses")
    for i, (r, source, text) in enumerate(tests, 1):
        a = classify(args.provider, model, args.mode, text, cache, delay)
        ok = "✓" if a["label"] == r["label"] else "✗"
        print(f"{i:3d}/{len(tests)} {ok} {r['id']:7s} {source:10s} gold={r['label']:7s} pred={a['label']:7s} {a['category']}")
        rows.append({"id": r["id"], "source": source, "difficulty": r["difficulty"],
                     "gold_category": r["category"], "gold_label": r["label"], "gold_severity": r["severity"],
                     "pred_category": a["category"], "pred_label": a["label"], "pred_severity": a["severity"],
                     "repaired": a.get("repaired", False), "reason": a["reason"]})

    res = pd.DataFrame(rows)
    name = f"llm_{args.provider}_{model.replace(':', '-').replace('/', '-')}_{args.mode}_{PROMPT_VERSION}"
    res.to_csv(f"results/{name}.csv", index=False)
    summarize(res, name)

    wrong = res[(res["source"] == "clean_text") & (res["gold_label"] != res["pred_label"])]
    if len(wrong):
        print("\nWrong problem/fine calls (clean text):")
        for _, w in wrong.iterrows():
            print(f"  {w['id']:7s} [{w['difficulty']}] gold={w['gold_label']:7s} pred={w['pred_label']:7s} — {w['reason']}")
    print(f"\nAll predictions → results/{name}.csv")


def metrics(r):
    out = {"n": len(r)}
    for t in ("category", "label", "severity"):
        out[t] = (r[f"gold_{t}"] == r[f"pred_{t}"]).mean()
    for d in ("easy", "hard"):
        s = r[r["difficulty"] == d]
        out[f"label_{d}"] = (s["gold_label"] == s["pred_label"]).mean() if len(s) else float("nan")
    out["invalid"] = (r["pred_label"] == "invalid").mean()
    out["repaired"] = r["repaired"].mean() if "repaired" in r else 0.0
    return out


def summarize(res, name):
    print(f"\n=== {name} ===")
    for source in res["source"].unique():
        m = metrics(res[res["source"] == source])
        print(f"{source:10s} n={m['n']:3d} | category {m['category']:6.1%} | problem/fine {m['label']:6.1%} "
              f"(easy {m['label_easy']:.0%}, hard {m['label_hard']:.0%}) | severity {m['severity']:6.1%}"
              + (f" | invalid answers {m['invalid']:.0%}" if m["invalid"] else "")
              + (f" | off-format answers repaired {m['repaired']:.0%}" if m["repaired"] else ""))


def compare():
    import glob
    rows = []
    for path in sorted(glob.glob("results/*.csv")):
        name = os.path.basename(path)[:-4]
        if name == "baseline_predictions":
            name = "baseline_tfidf (leave-one-out)"
        res = pd.read_csv(path)
        if "gold_label" not in res:
            continue
        for source in res["source"].unique():
            if source == "ocr_clean":
                continue
            m = metrics(res[res["source"] == source])
            rows.append({"model": name, "tested_on": source, "n": m["n"],
                         "category": f"{m['category']:.1%}", "problem/fine": f"{m['label']:.1%}",
                         "pf_easy": f"{m['label_easy']:.0%}", "pf_hard": f"{m['label_hard']:.0%}",
                         "severity": f"{m['severity']:.1%}"})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv("results/comparison.csv", index=False)
    print("\n→ results/comparison.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(DEFAULTS), default="gemini")
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", choices=["zero", "law"], default="zero")
    ap.add_argument("--bank", default=None)
    ap.add_argument("--ocr-dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only N clauses (quick test)")
    ap.add_argument("--delay", type=float, default=None, help="seconds between calls (rate limits)")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt, don't call the API")
    ap.add_argument("--compare", action="store_true", help="print a table of every run so far")
    args = ap.parse_args()
    compare() if args.compare else run(args)


if __name__ == "__main__":
    main()
