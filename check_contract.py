"""
Renewal Shield — check one contract.

Photo (or text) of a Dubai tenancy contract  ->  OCR  ->  clauses + fields
->  hybrid LLM check of every clause (practice guidance + retrieved law articles)
->  a readable report: red first, then yellow, then fine.

This is the user-facing part. The evaluation scripts (ocr_pipeline.py, classify_llm.py)
measure how good each step is; this one just runs the best setup end to end.

Usage:
  python check_contract.py --image out/photo/contract_0001.png
  python check_contract.py --image my_contract.jpg --provider gemini
  python check_contract.py --text contract.txt                  # already-typed contract text
  python check_contract.py --clause "The deposit is non-refundable."   # one clause
  python check_contract.py --image my_contract.jpg --no-llm     # OCR only: check what was read

Every run writes reports/<name>.md (readable) and reports/<name>.json (for the web app later).
LLM answers are cached (results/llm_cache.jsonl), so checking the same contract twice is free.

Not legal advice: the report flags clauses to look at, it does not replace a lawyer or the RDC.
"""

import argparse
import json
import os
import sys
import time
from datetime import date

from classify_llm import DEFAULTS, PROMPT_VERSION, classify, load_cache

MODE = "hybrid"            # best setup on the benchmark (see RESULTS.md)

ICON = {"red": "🔴", "yellow": "🟡", "none": "✅", "invalid": "❔", "unchecked": "📄"}
TITLE = {
    "red": "Likely conflicts with the law",
    "yellow": "Legal, but worth negotiating",
    "none": "Looks fine",
    "invalid": "Could not check",
    "unchecked": "Read, not checked (--no-llm)",
}
ORDER = ["red", "yellow", "invalid", "none", "unchecked"]

DISCLAIMER = ("Not legal advice. This report flags clauses worth a closer look; it can be wrong. "
              "For a dispute, contact the Rental Dispute Center (RDC) at the Dubai Land Department or a lawyer.")


# ----------------------------------------------------------------- input -> clauses + fields
def read_image(path, engine="rapidocr"):
    from ocr_pipeline import process
    print(f"Reading {path} with {engine} (takes 10-20 s on a laptop) ...")
    r = process(path, engine)
    print(f"  {len(r['clauses'])} clauses found in {r['seconds']:.0f} s")
    return r["text"], r["clauses"], r["fields"]


def read_text(path):
    from ocr_pipeline import extract_fields, split_clauses
    text = open(path, encoding="utf-8").read()
    clauses = split_clauses(text)
    if not clauses:                                   # no "Additional Terms" heading: one clause per line
        clauses = [l.strip() for l in text.splitlines() if len(l.strip()) > 15]
    return text, clauses, extract_fields(text)


# ----------------------------------------------------------------- field checks (no LLM)
def _parse_date(s):
    try:
        return date.fromisoformat(s) if s else None
    except ValueError:
        return None


def check_fields(f):
    """Simple consistency checks on the numbers read from the form. Notes, not legal findings."""
    notes = []
    rent, value, dep = f.get("annual_rent"), f.get("contract_value"), f.get("security_deposit")
    if rent and value and rent != value:
        notes.append(f"Contract value (AED {value:,}) differs from the annual rent (AED {rent:,}). "
                     "Fine if the term isn't one year; otherwise check which figure is right.")
    if rent and dep and dep > 0.10 * rent:
        notes.append(f"Security deposit is {dep / rent:.0%} of the annual rent. 5% (unfurnished) or 10% (furnished) "
                     "is common practice in Dubai. Not a legal cap, but worth asking about.")
    start, end = _parse_date(f.get("start_date")), _parse_date(f.get("end_date"))
    if start and end:
        days = (end - start).days
        if days <= 0:
            notes.append("End date is before the start date: probably an OCR mistake, check the paper copy.")
        elif not 330 <= days <= 370:
            notes.append(f"Contract runs {days} days, not about a year. Check the dates.")
    mode, cheques = f.get("payment_mode"), f.get("num_cheques")
    if mode == "Bank Transfer" and cheques:
        notes.append(f"Payment mode is bank transfer but {cheques} cheques are listed.")
    if mode == "Cheque" and cheques == 0:
        notes.append("Payment mode is cheque but no cheques are listed.")
    return notes


# ----------------------------------------------------------------- clause checks
def check_clauses(clauses, provider, model, retriever, k):
    from law_index import retrieve
    default_model, env, delay = DEFAULTS[provider]
    model = model or default_model
    if env and not os.environ.get(env):
        raise SystemExit(f"{env} not found. Add a line  {env}=your-key  to the .env file in the project folder.")
    cache = load_cache()
    results = []
    for i, text in enumerate(clauses, 1):
        print(f"  checking clause {i}/{len(clauses)} ...", end="\r")
        arts = retrieve(text, k, retriever)
        ans = classify(provider, model, MODE, text, cache, delay, arts)
        results.append({
            "n": i, "text": text,
            "category": ans["category"], "label": ans["label"], "severity": ans["severity"],
            "reason": ans["reason"],
            "articles_checked": [f"{a['source']}, Art. {a['article']}" for a in arts],
        })
    print(" " * 40, end="\r")
    return results, model


# ----------------------------------------------------------------- report
def summary_line(results):
    counts = {s: sum(r["severity"] == s for r in results) for s in ORDER}
    parts = [f"{ICON[s]} {counts[s]} {TITLE[s].lower()}" for s in ORDER if counts[s]]
    return " · ".join(parts) if parts else "no clauses checked"


def to_markdown(report):
    out = [f"# Contract check: {report['name']}", "",
           f"*{report['checked_at']} · {report['provider']} / {report['model']} · prompt {report['prompt_version']}*", ""]
    res = report["clauses"]
    if res:
        out += [f"**{len(res)} clause{'s' * (len(res) != 1)}:** {summary_line(res)}", ""]

    for sev in ORDER:
        group = [r for r in res if r["severity"] == sev]
        if not group:
            continue
        out += [f"## {ICON[sev]} {TITLE[sev]} ({len(group)})", ""]
        for r in group:
            out.append(f"**Clause {r['n']}** · {r['category'].replace('_', ' ')}  ")
            out.append(f"> {r['text']}")
            out.append("")
            if sev not in ("none", "unchecked"):
                out.append(f"**Why:** {r['reason']}  ")
                out.append(f"*Law articles checked: {'; '.join(r['articles_checked'])}*")
                out.append("")

    if report["fields"]:
        out += ["## Contract details (as read, check against your copy)", "", "| Field | Value |", "|---|---|"]
        for k, v in report["fields"].items():
            if v is None:
                v = "— (not read)"
            elif k in ("annual_rent", "contract_value", "security_deposit"):
                v = f"AED {v:,}"
            out.append(f"| {k.replace('_', ' ')} | {v} |")
        out.append("")
    if report["field_notes"]:
        out += ["**Notes on the details:**", ""] + [f"- {n}" for n in report["field_notes"]] + [""]

    out += ["---", "", f"*{DISCLAIMER}*", ""]
    return "\n".join(out)


def print_report(report):
    res = report["clauses"]
    print()
    print("=" * 70)
    print(f" CONTRACT CHECK: {report['name']}")
    print(f" {len(res)} clause{'s' * (len(res) != 1)}: {summary_line(res)}")
    print("=" * 70)
    for sev in ORDER:
        group = [r for r in res if r["severity"] == sev]
        if not group:
            continue
        print(f"\n{ICON[sev]} {TITLE[sev].upper()} ({len(group)})")
        for r in group:
            text = r["text"] if len(r["text"]) <= 160 else r["text"][:157] + "..."
            print(f"\n  [{r['n']}] {text}")
            if sev not in ("none", "unchecked"):
                print(f"      Why: {r['reason']}")
    if report["field_notes"]:
        print("\nNOTES ON THE CONTRACT DETAILS")
        for n in report["field_notes"]:
            print(f"  - {n}")
    print("\n" + DISCLAIMER)


def save(report, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, report["name"])
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(to_markdown(report))
    return base + ".md", base + ".json"


# ----------------------------------------------------------------- main API (the web app will call this)
def check(image=None, text_file=None, clause=None, provider="openai", model=None,
          retriever="bm25", k=3, engine="rapidocr", use_llm=True):
    if image:
        name = os.path.splitext(os.path.basename(image))[0]
        text, clauses, fields = read_image(image, engine)
    elif text_file:
        name = os.path.splitext(os.path.basename(text_file))[0]
        text, clauses, fields = read_text(text_file)
    else:
        name, text, clauses, fields = "single_clause", clause, [clause], {}

    if not clauses:
        print("\nNo clauses found. The 'Additional Terms' section may be cut off or too blurry.\n"
              "Try a straighter, brighter photo of the whole page, or run with --no-llm to see what was read.")

    report = {
        "name": name, "checked_at": time.strftime("%Y-%m-%d %H:%M"),
        "provider": provider, "model": model or DEFAULTS[provider][0],
        "prompt_version": PROMPT_VERSION, "mode": MODE, "retriever": f"{retriever}-k{k}",
        "fields": {k_: v for k_, v in fields.items()} if fields else {},
        "field_notes": check_fields(fields) if fields else [],
        "ocr_text": text,
        "clauses": [],
    }
    if use_llm and clauses:
        report["clauses"], report["model"] = check_clauses(clauses, provider, model, retriever, k)
    elif clauses:   # --no-llm: list what was read, unchecked
        report["clauses"] = [{"n": i, "text": c, "category": "-", "label": "-", "severity": "unchecked",
                              "reason": "", "articles_checked": []}
                             for i, c in enumerate(clauses, 1)]
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):             # Windows terminals: allow the emoji
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Check a Dubai tenancy contract against the law.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="photo or scan of the contract")
    src.add_argument("--text", help="contract text file")
    src.add_argument("--clause", help="a single clause, in quotes")
    ap.add_argument("--provider", default="openai", choices=list(DEFAULTS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--retriever", default="bm25", choices=["bm25", "openai"])
    ap.add_argument("-k", type=int, default=3, help="law articles given to the model per clause")
    ap.add_argument("--engine", default="rapidocr", choices=["rapidocr", "tesseract"])
    ap.add_argument("--no-llm", action="store_true", help="OCR only, no API calls")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    for p in (args.image, args.text):
        if p and not os.path.exists(p):
            raise SystemExit(f"File not found: {p}")

    report = check(args.image, args.text, args.clause, args.provider, args.model,
                   args.retriever, args.k, args.engine, use_llm=not args.no_llm)
    print_report(report)
    md, js = save(report, args.out)
    print(f"\nReport saved: {md}  (and {os.path.basename(js)})")


if __name__ == "__main__":
    main()
