"""Train a FUTO Keyboard typing model for one language, end to end.

Stages (default: all, in this order):
  download   public corpora from languages/<code>.yaml -> <work>/raw        (download.py)
  prepare    clean, SentencePiece tokenizer, tokenize -> <work>/tok         (prepare.py)
  train      35.5M Llama, LM + autocorrect examples   -> <work>/model       (train.py; CUDA recommended)
  evaluate   next-word / autocorrect top-1 and top-3                        (evaluate.py)
  export     FUTO GGUF, F16                            -> <work>/<code>_typing_f16.gguf   (export.py)
  quantize   llama-quantize to Q6_K                    -> <work>/<code>_typing_Q6_K.gguf  (install this one)

Every stage can be re-run alone (--stages train,export); each reads the previous stage's output from <work>.

Usage: python pipeline.py --lang xx [--work work/xx] [--stages all] [--tokens 1.5e9]
       python pipeline.py --lang sq --stages train,evaluate,export,quantize --init work/sq/model --error-tags v2 \
                          --tokens 3e8 --train-args "--lr 3e-4" --work work/sq-v2 --tok work/sq/tok   (a fine-tune)
"""
import argparse
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time

import langconfig

HERE = pathlib.Path(__file__).resolve().parent
STAGES = ["download", "prepare", "train", "evaluate", "export", "quantize"]

p = argparse.ArgumentParser()
p.add_argument("--lang", required=True, help="language code (languages/<code>.yaml) or config path")
p.add_argument("--work", default="", help="working directory (default: work/<code>)")
p.add_argument("--stages", default="all", help=f"comma-separated subset of: {','.join(STAGES)}")
p.add_argument("--tok", default="", help="use this tok dir instead of <work>/tok (e.g. to fine-tune on the same data)")
p.add_argument("--tokens", default="1.5e9", help="training tokens (1.5e9 = ~2h15m on an RTX 4090)")
p.add_argument("--init", default="", help="fine-tune from this model dir instead of training from scratch")
p.add_argument("--error-tags", default="", help="enable tagged typing_errors stages, e.g. v2")
p.add_argument("--train-args", default="", help='extra train.py arguments, e.g. "--batch 128 --workers 6"')
p.add_argument("--prepare-args", default="", help='extra prepare.py arguments, e.g. "--max-lines 5000"')
p.add_argument("--quant", default="Q6_K", help="comma-separated llama-quantize types")
p.add_argument("--llama-quantize", default=os.environ.get("LLAMA_QUANTIZE", ""), help="path to llama-quantize")
p.add_argument("--author", default="FUTO Keyboard Polyglot contributors")
args = p.parse_args()

cfg = langconfig.load(args.lang)
code = cfg["code"]
work = pathlib.Path(args.work or f"work/{code}")
raw, tok, model = work / "raw", pathlib.Path(args.tok) if args.tok else work / "tok", work / "model"
f16 = work / f"{code}_typing_f16.gguf"
stages = STAGES if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
assert all(s in STAGES for s in stages), f"unknown stage in {stages}"
work.mkdir(parents=True, exist_ok=True)
logf = open(work / "pipeline.log", "a", encoding="utf-8")
env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "TRANSFORMERS_VERBOSITY": "error"}


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True); logf.write(line + "\n"); logf.flush()


def run(script, *a):
    cmd = [sys.executable, str(HERE / script), *map(str, a)]
    log("$ " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        print(line, end="", flush=True); logf.write(line); logf.flush()
    if proc.wait():
        log(f"{script} failed (exit {proc.returncode}); stopping")
        sys.exit(proc.returncode)


def quantize():
    """llama-quantize if it's installed, else llama-cpp-python's copy of the same code; then check_gguf.py."""
    exe = args.llama_quantize or shutil.which("llama-quantize")
    try:
        import llama_cpp
    except ImportError:
        llama_cpp = None
    if not exe and not llama_cpp:
        log("llama-quantize not found. `pip install llama-cpp-python`, or install llama.cpp (e.g. `brew install "
            "llama.cpp`, or a release from github.com/ggml-org/llama.cpp/releases) and pass --llama-quantize. By hand:")
        for q in args.quant.split(","):
            log(f"  llama-quantize {f16} {work / f'{code}_typing_{q}.gguf'} {q}")
        return
    outs = []
    for q in args.quant.split(","):
        out = work / f"{code}_typing_{q}.gguf"
        if exe:
            r = subprocess.run([exe, str(f16), str(out), q], capture_output=True, text=True)
            rc, msg = r.returncode, r.stdout[-2000:] + r.stderr[-2000:]
        else:
            import ctypes
            qp = llama_cpp.llama_model_quantize_default_params()
            qp.ftype = getattr(llama_cpp, f"LLAMA_FTYPE_MOSTLY_{q}")
            rc, msg = llama_cpp.llama_model_quantize(str(f16).encode(), str(out).encode(), ctypes.byref(qp)), ""
        if rc:
            log(f"{q} failed:\n{msg}")
            sys.exit(rc)
        log(f"{out} {out.stat().st_size / 1e6:.1f} MB")
        outs.append(out)
    if llama_cpp:  # the real test of an export: llama.cpp's logits vs PyTorch's
        run("check_gguf.py", model, f16, *outs)
    else:
        log("tip: `pip install llama-cpp-python` and this stage also checks the files in llama.cpp (check_gguf.py)")


log(f"== {cfg['name']} ({code}): {', '.join(stages)} in {work}")
for stage in stages:
    log(f"== {stage}")
    if stage == "download":
        run("download.py", "--lang", args.lang, "--out", raw)
    elif stage == "prepare":
        run("prepare.py", "--lang", args.lang, "--raw", raw, "--tok", tok, *shlex.split(args.prepare_args))
    elif stage == "train":
        extra = ["--init", args.init] if args.init else []
        if args.error_tags:
            extra += ["--error-tags", args.error_tags]
        run("train.py", "--lang", args.lang, "--data", tok, "--out", model, "--tokens", args.tokens, *extra,
            *shlex.split(args.train_args))
    elif stage == "evaluate":
        run("evaluate.py", model, "--lang", args.lang, "--tok", tok,
            *(["--error-tags", args.error_tags] if args.error_tags else []))
    elif stage == "export":
        run("export.py", model, f16, "--lang", args.lang, "--author", args.author)
    elif stage == "quantize":
        quantize()
log("== all done")
