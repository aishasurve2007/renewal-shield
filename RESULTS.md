# Renewal Shield — Results (Phase 3)

Can an AI pipeline read a photo of a Dubai tenancy contract and flag clauses that conflict with
tenancy law, or are legal but bad for the tenant?

**Short answer:** on synthetic contracts, yes. The best setup (OCR → GPT-4.1-mini given both
practice guidance and the retrieved law text) labels clauses correctly 98% of the time, against
47% for a classic text classifier. The more interesting results are *why* each simpler setup fails.

> ⚠️ All numbers come from **synthetic data** that I generated, with labels I have not finished
> verifying. See [Limitations](#limitations) before quoting any number.

---

## 1. Task

For each clause in a contract's "Additional Terms" section, predict:

| Output | Values |
|---|---|
| **category** | rent_increase · notice_period · landlord_termination · early_termination · maintenance · deposit · other (from prompt v3) |
| **label** | problem · fine |
| **severity** | 🔴 red (likely conflicts with the law) · 🟡 yellow (legal but unfavourable, negotiate) · none |

**Law used:** Dubai Law No. 26 of 2007, *as amended by* Law No. 33 of 2008, plus Decree No. 43 of 2013 (rent increase caps).
The amendment replaced Articles 2, 3, 4, 9, 13, 14, 15, 25, 26, 29 and 36. Several of these
changes matter for tenants, e.g. eviction at expiry now needs **12 months'** notice (it was 90 days in 2007).

---

## 2. Data

| Component | Details |
|---|---|
| Clause bank | 60 clauses: 6 categories × (5 problem + 5 fine). 24 are "hard": tricky wording or *hard negatives* (look risky, are fine). |
| Contracts | Synthetic contracts built from the clause bank (`generate_contracts.py`): realistic fields, placeholder amounts filled per contract, consistent payment details. |
| Images | Each contract is rendered **clean** (scan-like) and as a **phone photo** (tilt, shadows, blur, noise, lower resolution). |
| Labels | Category, problem/fine, severity and law reference per clause, checked against the official law text (verification still in progress). |

---

## 3. OCR (contract photo → clauses + fields)

`ocr_pipeline.py`: RapidOCR + deskewing + retrying low-confidence lines + clause splitting + regex field extraction.

| Test set | Image | Clauses found | Character error | Fields correct |
|---|---|---|---|---|
| out: 10 contracts (development) | clean | 100% | 1.2% | 98.9% |
| out: 10 contracts (development) | photo | 100% | 2.6% | 100% |
| data2: 40 contracts (development†) | clean | 100% | 0.9% | 100% |
| data2: 40 contracts (development†) | photo | 100% | 2.0% | 99.4% |
| **data3: 40 contracts (held-out)** | **clean** | **100%** | **0.8%** | **98.9%** |
| **data3: 40 contracts (held-out)** | **photo** | **98.9%** | **2.8%** | **99.4%** |

†data2 started as a held-out set, but its failures were used to improve the clause splitter, so it
became a development set. data3 (seed 2026) was generated afterwards and never looked at while tuning:
**its numbers are the ones to quote.** Its only miss (contract 12, photo, 4/6) was deliberately left unfixed.

**What made the difference (each fix measured):**

| Fix | Effect |
|---|---|
| Straighten (deskew) the page before reading | Tilted photos stopped mixing words between lines. Also lifted Tesseract on photos from 31% → 82% clause recall. |
| Re-read low-confidence lines from a bigger crop | Long lines that came back as garbage ("ea") are read correctly. |
| Accept the clause numbers OCR actually produces ("1..", "2:", "3,", "4.7 The") | Clauses with misread numbers are split correctly instead of being lost. |
| Start a new clause after a short line when the number is missing | Two clauses no longer merge when OCR drops a clause number. |
| Tolerant section heading ("ADDITIONAL.TERMS", "ADD1TIONAL") | The splitter no longer turns field lines into fake clauses. |
| Repair missing spaces ("berefunded" → "be refunded") | Glued words are split back into real words; lower character error on photos. |
| Remove coloured ink (red watermark, blue signatures) | Little effect once deskewing was in place. Reported anyway. |

**Lesson:** each "held-out" set only stays held out until you fix something based on it. On data2,
photo recall was first 94.5%; that turned out to be partly a scoring bug (below) and partly two real
failures: OCR dropping a clause number ("3.") so two clauses merged, a misread heading
("ADDITIONAL.TERMS") and stray marks after numbers ("4.7 The..."). After fixing those, data2 reached
100%, but by then it was no longer an honest test. A fresh set (data3) confirmed the fixes generalise:
98.9% photo recall.

**Scoring note:** an earlier version of the scorer matched clauses in order, so one merged
clause could cascade into several false misses (contract 36 scored 1/6 instead of its true 4/6).
It now matches the best pairs first. All numbers above use the fixed scorer.

**Speed:** about 12–18 s per page on a laptop CPU. Fine for a benchmark, too slow for a user without a progress indicator.

---

## 4. Clause classification

Problem/fine and severity are the hard parts; category is easy for every LLM setup.

### 4.1 Results (clean clause text, n = 60)

| Model | What the model gets | Category | Problem/fine | Easy | Hard | Severity |
|---|---|---|---|---|---|---|
| TF-IDF + logistic regression | — (trained, leave-one-clause-out) | 78.3% | 46.7% | 42% | 54% | 36.7% |
| GPT-4.1-mini · zero | category descriptions only | 98.3% | 91.7% | 92% | 92% | 71.7% |
| GPT-4.1-mini · law | + practice guidance (my rule summaries) | 100% | 95.0% | 97% | 92% | 93.3% |
| GPT-4.1-mini · oracle | + the **correct** articles, verbatim | 98.3% | 95.0% | 94% | 96% | 68.3% |
| GPT-4.1-mini · rag (BM25) | + **retrieved** articles, verbatim | 96.7% | 95.0% | 94% | 96% | 71.7% |
| GPT-4.1-mini · rag (embeddings) | + retrieved articles, verbatim | 98.3% | 93.3% | 94% | 92% | 66.7% |
| GPT-4.1-mini · hybrid (BM25) | guidance + retrieved articles | 100% | 98.3% | 100% | 96% | 95.0% |
| GPT-4.1-mini · hybrid (embeddings) | guidance + retrieved articles | 100% | 95.0% | 97% | 92% | 90.0% |
| **GPT-4.1-mini · hybrid (BM25), prompt v3** | **+ "other" category, current-condition guidance** | **100%** | **98.3%** | **100%** | **96%** | **98.3%** |

All rows use prompt v2 except the last. v3 was written after the real-wording check (section 4.3);
on the benchmark it changes no problem/fine call and fixes two severity calls.

On clauses read by OCR from photos (n = 30), hybrid scores 96.7% problem/fine and 93–97% severity
(v3: 96.7% / 96.7%): **the OCR step costs essentially nothing.**

At n = 60, one clause = 1.7 points. Differences of one or two clauses (e.g. hybrid BM25 vs hybrid embeddings) are noise.

### 4.2 Retrieval (does the right article reach the model?)

| Retriever | hit@1 | hit@3 | hit@5 |
|---|---|---|---|
| BM25 (keywords) | 35% | 55% | 67% |
| OpenAI embeddings | 43% | 72% | 83% |

hit@3 by category:

| | Deposit | Maintenance | Rent increase | Notice | Landlord termination | Early termination |
|---|---|---|---|---|---|---|
| BM25 | 10/10 | 3/10 | 6/10 | 6/10 | 8/10 | **0/10** |
| Embeddings | 10/10 | **9/10** | 8/10 | 7/10 | 7/10 | **2/10** |

### 4.3 Real-wording check: the official Ejari standard terms

The benchmark clauses were all written by me. As a first check on real wording, I ran hybrid on the
**14 standard terms printed on the official DLD/Ejari unified tenancy contract**, copied word for word
from the DLD PDF. They are all lawful by definition (the regulator wrote them), so the only question is
whether the model raises **false alarms**. Most don't fit my six categories, so their gold category is "other".

| Prompt | False alarms | Category | Right article given |
|---|---|---|---|
| v2 (six categories, no "other") | 1/14 | 14.3% | 71% |
| **v3 ("other" + current-condition guidance)** | **0/14** | **92.9%** | 71% |

- **v2's false alarm (DLD-01):** *"The tenant has inspected the premises and agreed to lease the unit on
  its current condition."* Flagged yellow as if it shifted repairs to the tenant. It doesn't: Art. 16
  still makes the landlord responsible for maintenance. v3's guidance says accepting the current
  condition is fine unless the clause *also* removes the repair duty or the right to claim.
- **v2's category score** was low simply because there was no "other" option: the model had to force
  "use the premises for residential purposes only" into one of six boxes.
- **v3's one category miss (DLD-03),** no alterations without written approval, was put under
  maintenance. Defensible; my category definitions don't say where alterations belong.

**Caveat:** v3 was written *after* seeing v2's mistakes on these 14 terms, so this set is now a
development set too, and 0/14 is not an independent number. It shows the fix works, not that the
model generalises to real contracts. That still needs real, redacted contracts nobody has tuned on.

---

## 5. Findings

**1. Word matching can't judge legality.** The TF-IDF baseline gets categories right (78%) but
problem/fine at coin-flip level (47%). "Notify **thirty (30)** days before expiry" (problem) and
"notify **ninety (90)** days before expiry" (fine) share almost every word.

**2. LLMs know a lot, including law that is out of date.** Zero-shot, GPT-4.1-mini reaches 92%. But it
called "vacate with three months' notice" and "personal use, 90 days' notice by email" lawful. That
matches the **2007** law; the 2008 amendment requires 12 months via notary or registered mail.

**3. A summary of the law did not override that outdated knowledge; the verbatim law text did.**
With my one-line rule ("12 months' notice…") in the prompt, both clauses stayed wrong in every run.
With the actual text of Art. 25 in the prompt (oracle, rag, hybrid), both were corrected. This is a
**knowledge conflict**: the model trusts its training over a paraphrase, but yields to the primary source.

**4. Law text alone makes the model over-strict.** With only verbatim articles, severity dropped from
93% to about 70%, and fine clauses were flagged: the tenant doing minor repairs ("not explicitly
supported by law"), even **120 days'** notice ("exceeds the 90 days required"). The law says what is
*allowed*, not what is *normal*.

**5. Combining both works best.** Hybrid (practice guidance + retrieved verbatim articles) keeps the
corrected law from finding 3 and the sense of what's normal from the guidance: 98.3% problem/fine, 95% severity.

**6. Retrieval quality mattered less than expected, and fails in a specific way.** BM25 retrieved the
right article only 55% of the time, yet scored the same as the oracle, because the model already knows
most of the rules; retrieval mainly matters for the few facts it has *wrong* (Art. 25 was retrieved for those).
Both retrievers fail on early termination: clauses say "penalty" and "leave early", while Art. 7 says
"may not be unilaterally terminated". Embeddings fix vocabulary gaps like "repairs" vs
"maintenance works" (3/10 → 9/10) but not this one.

**7. Missing an "other" option breaks real contracts.** Real contracts are mostly boilerplate that fits
none of the six risk categories. Without "other", the model forced every clause into a category and once
invented a problem to go with it (section 4.3).

**Remaining errors in the best run (same in v2 and v3):**
- **NP-F04**, the tenant's own 60-day non-renewal notice, labelled "fine" but predicted "problem". This
  may be a labelling question: the 2007 Art. 14 covered non-renewal *and* amendments, while the 2008
  version only covers amendments. To be settled during label verification.
- **DE-F05** read from a photo fails in every mode, while its clean text never does: likely an OCR
  clause-splitting issue (two clauses merged).

---

## 6. Limitations

- **Synthetic data.** The clauses and the labels were written by the same person/process. Real contracts
  have wording, layouts and mistakes this data doesn't. A held-out set of real (redacted) contracts is the next step.
- **Labels not fully verified.** Law references were checked against the official texts; the full
  per-clause verification is still in progress (see NP-F04).
- **Small n.** 60 clauses, 30 photo clauses. One clause = 1.7 points.
- **Prompt tuned on the test set.** Prompt v2 (an explicit "yellow/red = problem" rule) was written
  after seeing v1's errors, and v3 after seeing v2's errors on the Ejari terms. All versions are
  reported. Only a new test set gives a clean number.
- **Partly circular guidance.** The practice guidance in the `law`/`hybrid` prompts was written alongside
  the labels, so it is close to a labelling guide. The `zero` and `oracle` modes don't use it.
- **One LLM.** GPT-4.1-mini only so far. Gemini runs are pending (free-tier quota).
- **Not legal advice.** The tool flags clauses to check; it doesn't replace a lawyer or the RDC.

---

## 7. Reproduce

```bash
pip install -r requirements.txt
python generate_contracts.py --n 10                        # synthetic contracts → out/
python ocr_pipeline.py --data out                          # OCR + evaluation
python generate_contracts.py --n 40 --out data2 --seed 999  # development set
python generate_contracts.py --n 40 --out data3 --seed 2026 # held-out set: don't tune on it
python ocr_pipeline.py --data data3
python ocr_pipeline.py --data data2 --rescore               # re-score saved OCR without re-running it
python classify_baseline.py --ocr-dir out                  # TF-IDF baseline (leave-one-clause-out)
python law_index.py --build                                # law PDFs → consolidated article index
python law_index.py --eval                                 # retrieval hit@k (add --retriever openai)
python classify_llm.py --provider openai --mode zero
python classify_llm.py --provider openai --mode law    --ocr-dir out
python classify_llm.py --provider openai --mode oracle --ocr-dir out
python classify_llm.py --provider openai --mode rag    --ocr-dir out
python classify_llm.py --provider openai --mode hybrid --ocr-dir out
python classify_llm.py --provider openai --mode hybrid --clauses real/real_clauses.csv  # Ejari terms
python classify_llm.py --compare                           # → results/comparison.csv
```

API keys go in `.env` (see `.env.example`). LLM answers are cached in `results/llm_cache.jsonl`,
so re-running reproduces the exact numbers above without new API calls.

## 8. Next

- Verify every label; settle NP-F04.
- Second model (Gemini) for the same table.
- 20–50 real redacted contracts as an untouched test set; run hybrid **once** on it.
- Deploy: FastAPI + hybrid pipeline behind an upload page.
