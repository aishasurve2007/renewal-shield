"""
Renewal Shield — Phase 3, step 2: baseline clause classifier.

For each clause it predicts:
  category : rent_increase / notice_period / landlord_termination /
             early_termination / maintenance / deposit
  label    : problem / fine
  severity : red / yellow / none

Model: TF-IDF (word 1-2 grams + character 3-5 grams, which are robust to OCR
typos) + logistic regression. Trains in seconds, no GPU, no API key.

HONEST EVALUATION — leave-one-clause-out:
The synthetic contracts reuse the same ~60 clause wordings, so a random split
of contracts would put the same sentence in train and test and score a fake
~100%. Instead, for every clause in the bank we train on all the OTHER clauses
and test on the one held out — the model never sees the wording it is tested on.
We test each held-out clause twice: its clean text, and (if available) the
versions our OCR actually read from contract photos.

Usage:
  python classify_baseline.py                       # evaluate on clean clause text
  python classify_baseline.py --ocr-dir out         # also test on OCR'd text
                                                    # (run ocr_pipeline.py --data out first)
  python classify_baseline.py --train               # train on everything, save model
  python classify_baseline.py --image out/photo/contract_0001.png   # end-to-end demo
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import FeatureUnion, make_pipeline

TASKS = ["category", "label", "severity"]
MODEL_PATH = "models/baseline_tfidf.joblib"


# ----------------------------------------------------------------- data
def load_bank(path=None):
    path = path or sorted(glob.glob("renewal_shield_clause_bank_*.xlsx"))[-1]
    df = pd.read_excel(path, sheet_name="Clause Bank").fillna("")
    df = df[df["verified"].astype(str).str.lower() != "no"].copy()
    df["text"] = [e.strip() if str(e).strip() else c.strip()
                  for c, e in zip(df["clause_text"], df["edited_text"])]
    # placeholders like {deposit} -> a neutral amount so the model sees normal text
    df["text"] = df["text"].str.replace(r"\{[a-z_]+\}", "AED 5,000", regex=True)
    print(f"Clause bank: {path} ({len(df)} clauses)")
    return df.reset_index(drop=True)


def load_ocr_variants(ocr_dir, engine="rapidocr"):
    """clause_id -> list of texts our OCR actually produced for that clause."""
    import difflib
    out = defaultdict(list)
    for lp in glob.glob(os.path.join(ocr_dir, "labels", "*.json")):
        gold = json.load(open(lp, encoding="utf-8"))
        for kind in ("photo", "clean"):
            op = os.path.join(ocr_dir, "ocr", engine, f"{gold['contract_id']}_{kind}.json")
            if not os.path.exists(op):
                continue
            pred = json.load(open(op, encoding="utf-8"))["clauses"]
            for c in gold["clauses"]:
                best = max(pred, default="", key=lambda p: difflib.SequenceMatcher(
                    None, p.lower(), c["text"].lower()).ratio())
                if best and difflib.SequenceMatcher(None, best.lower(), c["text"].lower()).ratio() > 0.6:
                    out[c["clause_id"]].append((kind, best))
    return out


# ----------------------------------------------------------------- model
def make_model():
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])
    return make_pipeline(features, LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced"))


def fit_all(df):
    return {t: make_model().fit(df["text"], df[t]) for t in TASKS}


def predict(models, texts):
    return {t: models[t].predict(texts) for t in TASKS}


# ----------------------------------------------------------------- evaluation
def evaluate(df, ocr_variants=None):
    rows = []
    for i in range(len(df)):
        held = df.iloc[i]
        train = df.drop(index=i)
        models = fit_all(train)
        tests = [("clean_text", held["text"])]
        for kind, t in (ocr_variants or {}).get(held["id"], []):
            tests.append((f"ocr_{kind}", t))
        for source, text in tests:
            p = predict(models, [text])
            rows.append({
                "id": held["id"], "source": source, "difficulty": held["difficulty"],
                **{f"gold_{t}": held[t] for t in TASKS},
                **{f"pred_{t}": p[t][0] for t in TASKS},
            })
    return pd.DataFrame(rows)


def report(res):
    print("\n=== LEAVE-ONE-CLAUSE-OUT RESULTS (model never saw the test wording) ===")
    for source in res["source"].unique():
        r = res[res["source"] == source]
        print(f"\n--- tested on {source} ({len(r)} predictions) ---")
        for t in TASKS:
            acc = (r[f"gold_{t}"] == r[f"pred_{t}"]).mean()
            f1 = f1_score(r[f"gold_{t}"], r[f"pred_{t}"], average="macro")
            print(f"{t:9s} accuracy {acc:6.1%}   macro-F1 {f1:.3f}")
        for diff in ["easy", "hard"]:
            d = r[r["difficulty"] == diff]
            if len(d):
                acc = (d["gold_label"] == d["pred_label"]).mean()
                print(f"  problem/fine on {diff:4s} clauses: {acc:6.1%} ({len(d)})")

    r = res[res["source"] == "clean_text"]
    print("\n--- problem/fine detail (clean text) ---")
    print(classification_report(r["gold_label"], r["pred_label"], digits=3, zero_division=0))
    wrong = r[r["gold_label"] != r["pred_label"]]
    if len(wrong):
        print("Misclassified (problem/fine):")
        for _, w in wrong.iterrows():
            print(f"  {w['id']:7s} [{w['difficulty']}] gold={w['gold_label']:7s} pred={w['pred_label']}")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=None)
    ap.add_argument("--ocr-dir", default=None, help="folder with labels/ and ocr/ from ocr_pipeline.py")
    ap.add_argument("--train", action="store_true", help="train on all clauses and save the model")
    ap.add_argument("--image", default=None, help="end-to-end: OCR an image and classify its clauses")
    args = ap.parse_args()

    if args.image:
        import ocr_pipeline
        models = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else fit_all(load_bank(args.bank))
        res = ocr_pipeline.process(args.image, "rapidocr")
        p = predict(models, res["clauses"]) if res["clauses"] else {t: [] for t in TASKS}
        icon = {"red": "[RED]   ", "yellow": "[YELLOW]", "none": "[OK]    "}
        print(f"Rent AED {res['fields']['annual_rent']} | deposit AED {res['fields']['security_deposit']} "
              f"| {res['fields']['start_date']} -> {res['fields']['end_date']}\n")
        for i, c in enumerate(res["clauses"]):
            sev = p["severity"][i] if p["label"][i] == "problem" else "none"
            print(f"{icon.get(sev, sev)} {p['category'][i]:21s} {c}")
        return

    df = load_bank(args.bank)

    if args.train:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(fit_all(df), MODEL_PATH)
        print(f"Saved → {MODEL_PATH}")
        return

    variants = load_ocr_variants(args.ocr_dir) if args.ocr_dir else None
    if variants:
        print(f"OCR variants found for {len(variants)} clause ids")
    res = evaluate(df, variants)
    os.makedirs("results", exist_ok=True)
    res.to_csv("results/baseline_predictions.csv", index=False)
    report(res)
    print("\nAll predictions → results/baseline_predictions.csv")


if __name__ == "__main__":
    main()
