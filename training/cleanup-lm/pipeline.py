"""One entry point for the whole cleanup-LM pipeline. Each step is a script you can also run alone:

  sources  sample_sources.py      work/sources_<code>.txt (skipped if present; --opus streams OpenSubtitles)
  pairs    gen_pairs.py           work/pairs_<code>.jsonl, rambler targets from the teacher server
  augment  augment_corrections.py work/pairs_corrections.jsonl
  train    train_cleanup.py       work/out/cleanup (merged) + work/out/cleanup-adapter (LoRA)   [CUDA GPU]
  eval     eval_cleanup.py        per-language scores on held-out pairs
  export   export_gguf.py         work/cleanup-q4_0.gguf, the file the app imports

Usage: python pipeline.py --langs el,sq,en --teacher-url http://localhost:8080/v1 --llama-cpp ../llama.cpp
       python pipeline.py --langs pt --opus --steps sources,pairs      (then read work/pairs_pt.jsonl before training)
"""
import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
STEPS = ["sources", "pairs", "augment", "train", "eval", "export"]


def run(*args):
    cmd = [sys.executable, *map(str, args)]
    print("\n>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", required=True, help="comma-separated codes with a languages/<code>.json")
    ap.add_argument("--work", default="work")
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--n", type=int, default=8000, help="source messages per language")
    ap.add_argument("--opus", action="store_true", help="sources step: sample from OPUS OpenSubtitles")
    ap.add_argument("--teacher", choices=["openai", "mlx"], default="openai")
    ap.add_argument("--teacher-url", default="http://localhost:8080/v1")
    ap.add_argument("--teacher-model", default="")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--llama-cpp", default="", help="llama.cpp checkout (export step)")
    a = ap.parse_args()

    langs, steps = a.langs.split(","), a.steps.split(",")
    work = pathlib.Path(a.work).resolve(); work.mkdir(parents=True, exist_ok=True)
    out = work / "out" / "cleanup"

    if "sources" in steps:
        for code in langs:
            src = work / f"sources_{code}.txt"
            if src.exists():
                print(f"{src.name} exists, keeping it", flush=True)
            elif a.opus:
                run("sample_sources.py", "--lang", code, "--out", src, "--n", a.n, "--opus")
            else:
                raise SystemExit(f"missing {src}: create it with sample_sources.py --source ..., or pass --opus")
    if "pairs" in steps:
        for code in langs:
            run("gen_pairs.py", "--lang", code, "--sources", work / f"sources_{code}.txt", "--out",
                work / f"pairs_{code}.jsonl", "--n", a.n, "--teacher", a.teacher, "--url", a.teacher_url,
                "--model", a.teacher_model, "--workers", a.workers, *(["--no-think"] if a.no_think else []))
    if "augment" in steps:
        run("augment_corrections.py", "--work", work)
    if "train" in steps:
        run("train_cleanup.py", "--work", work, "--out", out, "--langs", a.langs, "--epochs", a.epochs,
            "--rank", a.rank)
    if "eval" in steps:
        run("eval_cleanup.py", "--work", work, "--langs", a.langs, "--hard", out)
    if "export" in steps:
        if not a.llama_cpp:
            raise SystemExit("export needs --llama-cpp <llama.cpp checkout>")
        run("export_gguf.py", "--llama-cpp", os.path.abspath(a.llama_cpp), "--model", out,
            "--out", work / "cleanup-q4_0.gguf")


if __name__ == "__main__":
    main()
