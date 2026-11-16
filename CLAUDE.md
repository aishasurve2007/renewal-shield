# CLAUDE.md: Renewal Shield

Photo of a Dubai tenancy contract → OCR → clause split → law retrieval → LLM → red / yellow / fine per clause.
Full numbers and findings: `RESULTS.md`. The README is the public summary.

## Stack
Python 3.11 · FastAPI (`app.py`, `static/index.html`) · RapidOCR **1.2.3 (pinned, do not upgrade)** ·
BM25 + OpenAI-embedding retrieval (`law_index.py`) · LLM classifier (`classify_llm.py`: openai / gemini / anthropic / ollama) ·
Docker → GitHub Actions → Azure Container Apps.

## Key files
| File | Role |
|---|---|
| `ocr_pipeline.py` | Photo → clauses + fields (deskew, orientation, retry weak lines, splitter) |
| `law_index.py` | Consolidated law (2008 amendments replace 2007 text), `law/articles.json`, retrieval + `--eval` hit@k |
| `classify_llm.py` | Prompt modes `zero · law · rag · oracle · hybrid`, prompt versions, cache, `--compare` |
| `check_contract.py` | One contract end to end → Markdown + JSON report (used by the app) |
| `app.py`, `demo_guard.py` | Web app + rate limits / Turnstile for the public demo |
| `renewal_shield_clause_bank_v0.3.xlsx` | 60 labelled clauses: the benchmark ground truth |
| `results/` | Predictions per run, `llm_cache.jsonl`, `comparison.csv` |

## Run
```bash
pip install -r requirements.txt && cp .env.example .env     # keys in .env
python law_index.py --build                                  # law index
python law_index.py --eval [--retriever openai]              # retrieval hit@k
python classify_llm.py --provider openai --mode hybrid --ocr-dir out
python classify_llm.py --compare                             # → results/comparison.csv
python check_contract.py --image out/photo/contract_0001.png
uvicorn app:app --reload                                     # http://127.0.0.1:8000
pip install -r requirements-test.txt && pytest               # tests: no API, ~2 s
python eval/run_api_evals.py                                 # all API-dependent evals in one go
```
Use `--dry-run` / `--limit 5` on `classify_llm.py` before any full API run.

## Rules
- **Minimal, targeted changes.** Give complete file rewrites when a file changes, not partial snippets.
- **Don't break what works.** Before and after any change to OCR, retrieval or prompts, re-run the relevant
  eval and report before → after. A regression is a blocker, not a footnote.
- **Benchmark honesty is the point of this project.**
  - New prompt = new version (v4, v5…). Never edit v2/v3 in place; old cached results must stay reproducible.
    Select with `--prompt-version`; a test pins the v3 prompt hash. Same for retrievers: add a new one, don't change `bm25`.
  - `eval/retrieval_heldout.csv` is held-out for retrieval: never tune topic lines (`law/topics.json`) on its misses.
  - Never delete or rewrite `results/llm_cache.jsonl` entries.
  - `data3` and any future real-contract set are **held-out**: never tune on them. If something was tuned on a set, say so in RESULTS.md.
  - Report every run, including the ones that got worse. n = 60, so 1 clause = 1.7 points: call 1–2 clause differences noise.
- Keep the RapidOCR pin. Keep `requirements-app.txt` small (deployed image).
- Changes to the files listed in `.github/workflows/build-image.yml` trigger an image build on push: mention it when touching them.
- Update RESULTS.md and the README status list when a result or feature changes. Not legal advice: keep that caveat.
- Never commit `.env` or API keys.
