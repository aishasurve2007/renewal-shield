"""
Renewal Shield — real-camera test.

Score real phone photos of printed synthetic contracts against their labels.
You don't need to say which contract each photo shows: the script finds the label file whose
clauses best match what the OCR read (searching every */labels folder), then scores:

  clauses found   gold clauses read correctly (similarity >= 0.8, same rule as ocr_pipeline.py)
  extra           junk "clauses" that aren't in the contract (signature lines, watermark ...)
  fields          rent, deposit, dates ... read correctly
  problem/fine    (with the LLM) correct calls on the clauses that were found
  severity        (with the LLM) correct red / yellow / none

Usage:
  python eval_phone.py --dir phone --no-llm        # OCR only, free
  python eval_phone.py --dir phone                 # full check (cached answers are free)

Name photos after their condition if you like (c1_good.jpg, c1_angle.jpg ...): the name is
kept in the results so you can compare conditions.
Results -> results/phone_eval.csv
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd

from check_contract import check
from ocr_pipeline import FIELD_PATTERNS, match_clauses

IMG = (".jpg", ".jpeg", ".png", ".webp")


def load_all_labels(pattern):
    out = []
    for path in sorted(glob.glob(pattern)):
        try:
            out.append((path, json.load(open(path, encoding="utf-8"))))
        except Exception:
            pass
    return out


def find_contract(pred_clauses, fields, labels):
    """Label file whose clauses best match the OCR'd clauses (fields break ties)."""
    best, best_score = None, 0.0
    for path, gold in labels:
        g = [c["text"] for c in gold["clauses"]]
        if not g or not pred_clauses:
            continue
        pairs = match_clauses(pred_clauses, g)
        score = sum(s for _, _, s in pairs) / len(g)
        score += 0.05 * sum(1 for k in ("annual_rent", "unit_no", "security_deposit")
                            if fields.get(k) is not None and fields.get(k) == gold["fields"].get(k))
        if score > best_score:
            best, best_score = (path, gold), score
    return best if best_score >= 0.5 else None


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="phone")
    ap.add_argument("--labels", default="*/labels/*.json", help="where to look for label files")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--out", default="results/phone_eval.csv")
    args = ap.parse_args()

    images = sorted(f for f in glob.glob(os.path.join(args.dir, "*")) if f.lower().endswith(IMG))
    if not images:
        raise SystemExit(f"No photos in {args.dir}/ (jpg/png).")
    labels = load_all_labels(args.labels)
    print(f"{len(images)} photos, {len(labels)} label files to match against\n")

    rows, clause_rows = [], []
    for img in images:
        name = os.path.basename(img)
        r = check(image=img, provider=args.provider, use_llm=not args.no_llm)
        pred = [c["text"] for c in r["clauses"]]
        hit = find_contract(pred, r["fields"], labels)
        if not hit:
            print(f"  {name}: could not match to any contract (OCR too poor) -> counted as a failure\n")
            rows.append({"photo": name, "contract": "?", "clauses_found": 0, "clauses_total": None,
                         "extra": len(pred), "fields_ok": 0, "fields_total": len(FIELD_PATTERNS)})
            continue
        path, gold = hit
        g = gold["clauses"]
        pairs = match_clauses(pred, [c["text"] for c in g])
        found = [(gc, p, s) for gc, (_, p, s) in zip(g, pairs) if s >= 0.8]
        fields_ok = sum(1 for k in FIELD_PATTERNS if r["fields"].get(k) == gold["fields"].get(k))
        row = {"photo": name, "contract": f"{os.path.basename(os.path.dirname(os.path.dirname(path)))}/"
                                          f"{gold['contract_id']}",
               "clauses_found": len(found), "clauses_total": len(g),
               "extra": len(pred) - len(found), "fields_ok": fields_ok, "fields_total": len(FIELD_PATTERNS)}

        if not args.no_llm:
            by_text = {c["text"]: c for c in r["clauses"]}
            pf = sev = 0
            for gc, p, s in found:
                pc = by_text[p]
                pf += pc["label"] == gc["label"]
                sev += pc["severity"] == gc["severity"]
                clause_rows.append({"photo": name, "clause_id": gc["clause_id"], "gold": gc["severity"],
                                    "pred": pc["severity"], "ok": pc["severity"] == gc["severity"]})
            row.update({"pf_ok": pf, "sev_ok": sev})
        rows.append(row)
        extra = f", {row['extra']} extra" if row["extra"] else ""
        llm = f" | problem/fine {row['pf_ok']}/{len(found)}, severity {row['sev_ok']}/{len(found)}" \
            if not args.no_llm else ""
        print(f"  {name}: {row['contract']}  clauses {len(found)}/{len(g)}{extra}  "
              f"fields {fields_ok}/{len(FIELD_PATTERNS)}{llm}\n")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    if clause_rows:
        pd.DataFrame(clause_rows).to_csv(args.out.replace(".csv", "_clauses.csv"), index=False)

    known = df[df["contract"] != "?"]
    perfect = ((known["clauses_found"] == known["clauses_total"]) & (known["extra"] == 0)).sum()
    print("=" * 70)
    print(f"Photos fully read (all clauses, no junk): {perfect}/{len(df)}")
    print(f"Photos that couldn't be matched at all:   {(df['contract'] == '?').sum()}/{len(df)}")
    if len(known):
        print(f"Clause recall (matched photos):           {known['clauses_found'].sum()}/{int(known['clauses_total'].sum())}"
              f" = {known['clauses_found'].sum() / known['clauses_total'].sum():.1%}")
        print(f"Field accuracy (matched photos):          {known['fields_ok'].sum()}/{known['fields_total'].sum()}"
              f" = {known['fields_ok'].sum() / known['fields_total'].sum():.1%}")
    if not args.no_llm and len(known):
        n = known["clauses_found"].sum()
        print(f"Problem/fine on clauses found:            {int(known['pf_ok'].sum())}/{n} = {known['pf_ok'].sum() / n:.1%}")
        print(f"Severity on clauses found:                {int(known['sev_ok'].sum())}/{n} = {known['sev_ok'].sum() / n:.1%}")
        wrong = [c for c in clause_rows if not c["ok"]]
        for c in wrong:
            print(f"  wrong: {c['photo']} {c['clause_id']} gold={c['gold']} pred={c['pred']}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
