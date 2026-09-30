"""One command from data to a FUTO-ready .bin: prepare -> train -> acft -> convert -> eval -> card.

Each step is a separate script run in its own process (so GPU memory is freed between steps) and is skipped when its
output already exists, so a rerun resumes where it stopped; --force redoes the listed steps.

  work/<code>/data/                 manifests + sources.json (shared by all sizes)
  work/<code>/<size>/model/         fine-tuned Transformers model (+ train_info.json)
  work/<code>/<size>/model-acft/    after ACFT
  work/<code>/<size>/<Name>-<N>.bin FUTO file (q8_0), next to its f16 source
  work/<code>/<size>/results.json   FUTO-mode scores
  work/<code>/<size>/card/          README.md + ATTRIBUTION.md
  work/<code>/<size>/pipeline.log

Example: python pipeline.py --lang el --size small --cv data/cv-corpus-27.0-2026-09-11/el --fleurs --whisper-cpp ../whisper.cpp
"""
import argparse
import datetime
import pathlib
import shlex
import subprocess
import sys

from common import HERE, SIZES, load_lang, read_jsonl

LR = {"small": 1.25e-5, "base": 2.5e-5, "tiny": 3.75e-5}   # smaller models want a higher rate
STEPS = ["prepare", "train", "acft", "convert", "eval", "card"]


def auto_epochs(hours):
    # Greek (~100 h) peaked around 4 epochs; Albanian (~8 h) was still improving slowly at 10.
    return 10 if hours < 15 else 6 if hours < 50 else 4


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, help="language code (languages/<code>.json) or config path")
    ap.add_argument("--size", default="small", choices=SIZES)
    ap.add_argument("--work", default="work")
    ap.add_argument("--steps", default=",".join(STEPS), help=f"comma list from {','.join(STEPS)}")
    ap.add_argument("--force", action="store_true", help="redo the selected steps even if their output exists")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    # data
    ap.add_argument("--cv", help="extracted Common Voice locale folder")
    ap.add_argument("--fleurs", action="store_true")
    ap.add_argument("--extra", action="append", default=[], help="extra training manifest (repeatable)")
    ap.add_argument("--extra-test", action="append", default=[], help="extra test manifest (repeatable)")
    ap.add_argument("--verify", action="append", default=[], help="extra training manifest checked by the teacher")
    ap.add_argument("--verify-other", action="store_true", help="add CV other.tsv clips the teacher confirms")
    ap.add_argument("--label-teacher", help="teacher for label checks (default: the language config's)")
    # training
    ap.add_argument("--base", help="starting model (default openai/whisper-<size>)")
    ap.add_argument("--teacher", default="", help="optional KD teacher, e.g. work/<code>/small/model for base/tiny")
    ap.add_argument("--epochs", type=float, default=0, help="default: 10 below 15 h of data, 6 below 50 h, else 4")
    ap.add_argument("--lr", type=float, default=0, help="default: small 1.25e-5, base 2.5e-5, tiny 3.75e-5")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--no-ckpt", action="store_true", help="no gradient checkpointing: faster, needs ~18 GB for small")
    ap.add_argument("--acft-epochs", type=int, default=4)
    ap.add_argument("--acft-clips", type=int, default=4000)
    # conversion / evaluation / card
    ap.add_argument("--whisper-cpp", default="", help="whisper.cpp checkout or bin folder")
    ap.add_argument("--quantize", default="q8_0")
    ap.add_argument("--stock", help="FUTO's stock .bin of the same size, to compare against")
    ap.add_argument("--eval-clips", type=int, default=200)
    ap.add_argument("--author", default="")
    ap.add_argument("--license", default="mit")
    ap.add_argument("--device", default="auto")
    # smoke tests
    ap.add_argument("--limit", type=int, default=0, help="max clips per source (prepare)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop training after N steps")
    ap.add_argument("--eval-every", type=int, default=600)
    ap.add_argument("--dev-clips", type=int, default=500)
    a = ap.parse_args()

    lang = load_lang(a.lang)
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    bad = set(steps) - set(STEPS)
    if bad:
        sys.exit(f"unknown steps: {bad}")
    work = pathlib.Path(a.work) / lang.code
    data, run = work / "data", work / a.size
    model, acft = run / "model", run / "model-acft"
    binp = run / f"{lang.name.replace(' ', '')}-{SIZES[a.size]}.bin"
    results = run / "results.json"
    run.mkdir(parents=True, exist_ok=True)
    log = run / "pipeline.log"
    py = [sys.executable, "-u"]

    def say(msg):
        line = f"{datetime.datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        if not a.dry_run:
            with log.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def run_step(name, args, done):
        if name not in steps:
            return
        if done.exists() and not a.force:
            say(f"== {name}: {done} exists, skipping (--force to redo)")
            return
        cmd = py + [str(HERE / args[0])] + [str(x) for x in args[1:]]
        say(f"== {name}: {' '.join(shlex.quote(c) for c in cmd[1:])}")
        if a.dry_run:
            return
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace") as p, log.open("a", encoding="utf-8") as f:
            for line in p.stdout:
                sys.stdout.write(line); f.write(line)
        if p.returncode:
            say(f"{name} failed (exit {p.returncode}); see {log}")
            sys.exit(p.returncode)

    say(f"### {lang.name} ({lang.code}), whisper-{a.size}, steps {','.join(steps)}")

    prep = ["prepare_data.py", "--lang", a.lang, "--data", data, "--device", a.device]
    if a.cv:
        prep += ["--cv", a.cv]
    if a.fleurs:
        prep += ["--fleurs"]
    for flag, vals in (("--extra", a.extra), ("--extra-test", a.extra_test), ("--verify", a.verify)):
        for v in vals:
            prep += [flag, v]
    if a.verify_other:
        prep += ["--verify-other"]
    if a.label_teacher:
        prep += ["--teacher", a.label_teacher]
    if a.limit:
        prep += ["--limit", a.limit]
    if "prepare" in steps and not (a.cv or a.fleurs or a.extra or a.verify) and not (data / "sources.json").exists():
        sys.exit("prepare needs data: --cv, --fleurs and/or --extra (see README)")
    run_step("prepare", prep, data / "sources.json")

    epochs = a.epochs
    if not epochs:
        hours = sum(r.get("dur", 0) for p in data.glob("train_*.jsonl") for r in read_jsonl(p)) / 3600 if data.exists() else 0
        epochs = auto_epochs(hours)
        if "train" in steps:
            say("epochs: decided from the prepared data's hours" if a.dry_run and not hours
                else f"{hours:.1f} h of training data -> {epochs} epochs")
    train = ["train.py", "--lang", a.lang, "--data", data, "--out", model, "--base", a.base or f"openai/whisper-{a.size}",
             "--epochs", epochs, "--lr", a.lr or LR[a.size], "--batch", a.batch, "--accum", a.accum, "--device", a.device,
             "--eval-every", a.eval_every, "--dev-clips", a.dev_clips]
    if a.teacher:
        train += ["--teacher", a.teacher]
    if a.no_ckpt:
        train += ["--no-ckpt"]
    if a.max_steps:
        train += ["--max-steps", a.max_steps]
    run_step("train", train, model / "train_info.json")

    run_step("acft", ["acft.py", "--lang", a.lang, "--model", model, "--out", acft, "--data", data, "--epochs", a.acft_epochs,
                      "--max-clips", a.acft_clips, "--device", a.device], acft / "config.json")

    conv = ["convert_ggml.py", acft, binp, "--quantize", a.quantize]
    if a.whisper_cpp:
        conv += ["--whisper-cpp", a.whisper_cpp]
    run_step("convert", conv, binp)

    ev = ["evaluate.py", "--lang", a.lang, "--data", data, "--models", binp] + ([a.stock] if a.stock else []) + \
         ["--n", a.eval_clips, "--json", results]
    if a.whisper_cpp:
        ev += ["--whisper-cpp", a.whisper_cpp]
    run_step("eval", ev, results)

    card = ["model_card.py", "--lang", a.lang, "--size", a.size, "--bin", binp, "--model", model, "--data", data,
            "--out", run / "card", "--license", a.license]
    if results.exists() or a.dry_run:
        card += ["--results", results]
    if a.stock:
        card += ["--stock", pathlib.Path(a.stock).name]
    if a.author:
        card += ["--author", a.author]
    run_step("card", card, run / "card" / "README.md")
    if "card" in steps or "convert" in steps:
        say(f"done: {binp} -> FUTO Keyboard: Settings > Languages & Models > {lang.name} > Voice Input > Import")


if __name__ == "__main__":
    main()
