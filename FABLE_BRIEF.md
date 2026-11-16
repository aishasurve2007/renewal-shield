# Fable session brief: Renewal Shield

## Where it stands
- Hybrid (guidance + verbatim retrieved law), prompt v3: **98.3% problem/fine, 98.3% severity** on 60 synthetic clauses.
- OCR: 98.9% clause recall on held-out photos; 20/23 real phone photos fully read; 0/14 false alarms on Ejari terms.
- Deployed on Azure; image built by GitHub Actions.

## Known weak spots (from RESULTS.md)
1. **Early-termination retrieval is broken:** hit@3 is 0/10 (BM25) and 2/10 (embeddings). Clauses say "penalty / leave early", Art. 7 says "may not be unilaterally terminated".
2. **Retrieval is fragile under OCR noise:** LT-P03 lost Art. 25 on one photo and the model fell back on the 2007 law.
3. **No article citation in the output yet** (README to-do: prompt v4).
4. **One LLM only** (GPT-4.1-mini). `classify_llm.py` already supports anthropic and gemini.
5. NP-F04 label question; DE-F05 photo clause merge.

## Goals for this session (in order)
1. **Fix retrieval.** Hybrid BM25 + embeddings (fusion), and/or query expansion from the clause category
   (e.g. early_termination → "unilaterally terminated", Art. 7). Target: early-termination hit@3 well above 2/10,
   no category worse than now. Report the full hit@k table before → after.
2. **Prompt v4: cite the article.** The model returns the article it relied on. Add a **citation accuracy** metric
   against the clause bank's law reference. Re-run hybrid; v3 must stay reproducible.
3. **Second model.** Run the same table with `--provider anthropic` (and Gemini if quota allows). Add it to RESULTS.md.
4. **Tests + CI (needed for 90):** pytest suite for the clause splitter, field parser, `parse_answer` and
   law consolidation (2008 replaces 2007), using small fixtures, no API calls. Add a test job to GitHub Actions
   (separate from the image build).
5. If time is left: investigate DE-F05 (OCR clause merge), and make the app show the cited article.

Work on a branch and open a PR: pushing to main triggers an image build.

Out of scope today: Arabic contracts, collecting real contracts (needs data I have to gather myself).

## First prompt to paste
> Read CLAUDE.md, FABLE_BRIEF.md, RESULTS.md and the code in law_index.py and classify_llm.py.
> Goal 1 first: fix retrieval for early termination without hurting other categories.
> Before changing anything, give me a plan: what you'll change, which eval you'll run before and after, and what could regress.
