"""
Run every evaluation that needs the OpenAI API, in one go, and save the output.

    python eval/run_api_evals.py            # from the project root, with OPENAI_API_KEY in .env

What it runs (about 250 short gpt-4.1-mini calls plus a few embedding calls: well under 1 USD):
  1. retrieval hit@k for the embedding retriever and the new `fusion` retriever,
     on the clause bank and on the held-out clauses (eval/retrieval_heldout.csv)
  2. hybrid, prompt v3, with the new bm25-topics retriever   -> did better retrieval change any call?
  3. hybrid, prompt v4 (cites the article), bm25-topics       -> citation accuracy
  4. the same two on the official Ejari terms, if real/real_clauses.csv exists (false alarms must stay 0)
  5. the comparison table

Answers are cached in results/llm_cache.jsonl, so a second run costs nothing and gives the same numbers.
Everything printed is also written to eval/api_eval_output.txt: paste or commit that file.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "eval", "api_eval_output.txt")
PY = sys.executable
OCR = ["--ocr-dir", "out"] if os.path.isdir(os.path.join(ROOT, "out")) else []
HYBRID = [PY, "classify_llm.py", "--provider", "openai", "--mode", "hybrid", "--retriever", "bm25-topics"]

STEPS = [
    [PY, "law_index.py", "--eval", "--retriever", "openai"],
    [PY, "law_index.py", "--eval", "--retriever", "fusion"],
    [PY, "law_index.py", "--eval", "--retriever", "openai", "--clauses", "eval/retrieval_heldout.csv"],
    [PY, "law_index.py", "--eval", "--retriever", "fusion", "--clauses", "eval/retrieval_heldout.csv"],
    HYBRID + ["--prompt-version", "v3"] + OCR,
    HYBRID + ["--prompt-version", "v4"] + OCR,
]
if os.path.exists(os.path.join(ROOT, "real", "real_clauses.csv")):
    STEPS += [HYBRID + ["--prompt-version", v, "--clauses", "real/real_clauses.csv"] for v in ("v3", "v4")]
STEPS.append([PY, "classify_llm.py", "--compare"])


def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    with open(OUT, "w", encoding="utf-8") as log:
        for cmd in STEPS:
            line = "\n$ " + " ".join(["python"] + cmd[1:])
            print(line, flush=True)
            log.write(line + "\n")
            p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
            text = p.stdout + (("\n[stderr]\n" + p.stderr) if p.returncode else "")
            print(text, flush=True)
            log.write(text)
            if p.returncode:
                print(f"Stopped: that command failed (exit {p.returncode}). Answers so far are cached; fix and re-run.")
                sys.exit(p.returncode)
    print(f"\nDone. Output saved to {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
