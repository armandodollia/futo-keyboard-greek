"""Build the training / dev / test manifests for one language.

Sources (all optional except that something must be given):
  --cv DIR       an extracted Common Voice locale folder (the one with validated.tsv, test.tsv, dev.tsv and clips/),
                 from the Mozilla Data Collective. test.tsv -> eval_cv_test, dev.tsv -> eval_cv_dev, validated.tsv
                 minus every clip whose speaker or sentence is in test/dev -> train_cv. Without that exclusion the test
                 score is inflated: CV repeats sentences across speakers.
  --verify-other additionally keep other.tsv clips (recorded but not yet voted on) whose prompt the teacher recognises
                 (loose CER <= 0.10) -> train_cv_other. Needs --teacher; slow without a GPU.
  --fleurs       Google FLEURS (CC-BY-4.0) train+validation -> train_fleurs, test -> eval_fleurs_test, when the
                 language config names a FLEURS config. Audio is exported as 16 kHz wav under data/fleurs/.
  --extra F      your own manifest(s) for training (format in README.md). Copied to train_<name>.jsonl after checking
                 every row, adding "dur", applying the language's `clean` rules and dropping sentences that occur in
                 any test/dev set. A sidecar F.source.json ({"name", "url", "license", "use"}) credits it in the model card.
  --extra-test F your own held-out manifest(s) -> eval_<name>_test.jsonl (evaluation only, never trained on).
  --verify F     check an --extra manifest's labels with the teacher first (loose CER <= --verify-cer), e.g. subtitles
                 or TTS clips. Same flag format as --extra.

Output: <data>/{train,eval}_*.jsonl and <data>/sources.json (attribution for the model card).
"""
import argparse
import csv
import json
import pathlib
import random
import re
import sys

import numpy as np

from common import SR, load_audio, load_lang, pick_device, read_jsonl, write_jsonl, write_wav, cer

csv.field_size_limit(2 ** 31 - 1)
MAX_DUR = 29.5   # Whisper's window is 30 s; longer clips would be cut mid-sentence


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def common_voice(cv: pathlib.Path, lang, out: pathlib.Path, limit: int):
    durs = {}
    if (cv / "clip_durations.tsv").exists():
        durs = {r["clip"]: int(r["duration[ms]"]) / 1000 for r in read_tsv(cv / "clip_durations.tsv")}
    test, dev = read_tsv(cv / "test.tsv"), read_tsv(cv / "dev.tsv")
    held_speakers = {r["client_id"] for r in test + dev}
    held_sentences = {lang.norm(r["sentence"]) for r in test + dev}

    def row(r):
        return {"audio": str(cv / "clips" / r["path"]), "text": lang.clean(r["sentence"]), "dur": durs.get(r["path"], 0.0),
                "src": "cv", "speaker": r["client_id"][:16]}

    def allowed(r):
        return r["sentence"].strip() and r["client_id"] not in held_speakers and lang.norm(r["sentence"]) not in held_sentences

    cut = (lambda rows: rows[:limit]) if limit else (lambda rows: rows)
    write_jsonl(out / "eval_cv_test.jsonl", cut([row(r) for r in test]))
    write_jsonl(out / "eval_cv_dev.jsonl", cut([row(r) for r in dev]))
    train = cut([row(r) for r in read_tsv(cv / "validated.tsv") if allowed(r)])
    train = [r for r in train if r["dur"] <= MAX_DUR]
    write_jsonl(out / "train_cv.jsonl", train)
    have = {r["audio"] for r in train}
    other = []
    if (cv / "other.tsv").exists():
        other = cut([row(r) for r in read_tsv(cv / "other.tsv") if allowed(r) and str(cv / "clips" / r["path"]) not in have])
    version = re.search(r"cv-corpus-[\d.]+", str(cv.resolve()))
    return other, held_sentences, {
        "name": f"Common Voice ({version[0] + ', ' if version else ''}{lang.code})",
        "url": "https://commonvoice.mozilla.org", "license": "CC0-1.0",
        "use": "Training (validated clips; test/dev speakers and sentences excluded); test and dev splits for evaluation",
        "citation": "Ardila et al., Common Voice: A Massively-Multilingual Speech Corpus, LREC 2020"}


def fleurs(lang, out: pathlib.Path, limit: int):
    from datasets import load_dataset
    wav_dir = out / "fleurs"; wav_dir.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "test": []}
    for split, dest in (("train", "train"), ("validation", "train"), ("test", "test")):
        try:
            ds = load_dataset("google/fleurs", lang.fleurs, split=split, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001 - FLEURS is a loading-script dataset; newer `datasets` releases refuse it
            print(f"FLEURS unavailable ({type(e).__name__}: {e}); skipping. Pin datasets<4 to use it.", flush=True)
            return None
        for i, ex in enumerate(ds):
            if limit and i >= limit:
                break
            a = ex["audio"]["array"]
            p = wav_dir / f"{split}_{ex['id']}_{i}.wav"
            if not p.exists():
                write_wav(p, a)
            rows[dest].append({"audio": str(p), "text": lang.clean(ex["raw_transcription"]), "dur": round(len(a) / SR, 2),
                               "src": "fleurs"})
    write_jsonl(out / "train_fleurs.jsonl", [r for r in rows["train"] if r["dur"] <= MAX_DUR])
    write_jsonl(out / "eval_fleurs_test.jsonl", rows["test"])
    return {"name": f"FLEURS ({lang.fleurs})", "url": "https://huggingface.co/datasets/google/fleurs", "license": "CC-BY-4.0",
            "use": "Training (train + validation); test split for evaluation",
            "citation": "Conneau et al., FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech, SLT 2022"}


class Teacher:
    """Batch transcription with a large Whisper, used only to accept or reject existing labels."""

    def __init__(self, name, lang, device):
        import torch
        from transformers import GenerationConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer
        self.torch, self.lang, self.device = torch, lang, device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(name, dtype=dtype).to(device).eval()
        self.dtype = dtype
        # Community fine-tunes often ship without tokenizer/preprocessor files: fall back to the matching OpenAI model
        # (large-v3 family = 128 mel bins and 51,866 tokens).
        ref = "openai/whisper-large-v3" if self.model.config.vocab_size == 51866 else "openai/whisper-small"
        self.fe = WhisperFeatureExtractor(feature_size=self.model.config.num_mel_bins)
        try:
            self.tok = WhisperTokenizer.from_pretrained(name)
        except OSError:
            self.tok = WhisperTokenizer.from_pretrained(ref)
        try:
            GenerationConfig.from_pretrained(name)
        except OSError:
            self.model.generation_config = GenerationConfig.from_pretrained(ref)

    def __call__(self, rows, batch=16):
        rows = sorted(rows, key=lambda r: r.get("dur", 0))      # similar lengths per batch = less padding work
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            audio = []
            for r in chunk:
                try:
                    audio.append(load_audio(r["audio"])[: SR * 30])
                except Exception:  # noqa: BLE001 - unreadable clip: an empty input gets rejected below
                    audio.append(None)
            ok = [a if a is not None else np.zeros(SR // 10, np.float32) for a in audio]
            feats = self.fe(ok, sampling_rate=SR, return_tensors="pt").input_features
            with self.torch.no_grad():
                ids = self.model.generate(feats.to(self.device, self.dtype), language=self.lang.whisper_language,
                                          task="transcribe", max_new_tokens=220)
            for r, a, hyp in zip(chunk, audio, self.tok.batch_decode(ids, skip_special_tokens=True)):
                yield r, ("" if a is None else hyp.strip())
            if (i // batch) % 50 == 0:
                print(f"  teacher: {i + len(chunk)}/{len(rows)} checked", flush=True)


def verify(rows, teacher, lang, max_cer):
    kept = []
    for r, hyp in teacher(rows):
        c = cer(lang.loose(r["text"]), lang.loose(hyp))
        if hyp and c <= max_cer:
            kept.append(dict(r, teacher=hyp, teacher_cer=round(c, 3)))
    print(f"  kept {len(kept)}/{len(rows)} (loose CER <= {max_cer})", flush=True)
    return kept


def load_extra(path, lang, held_sentences, limit):
    rows = read_jsonl(path)[: limit or None]
    good = []
    for r in rows:
        if not pathlib.Path(r["audio"]).exists():
            print(f"  missing audio, skipped: {r['audio']}", flush=True); continue
        r["text"] = lang.clean(r["text"])
        if not r["text"] or lang.norm(r["text"]) in held_sentences:
            continue
        if not r.get("dur"):
            r["dur"] = round(len(load_audio(r["audio"])) / SR, 2)
        if r["dur"] <= MAX_DUR:
            r.setdefault("src", pathlib.Path(path).stem)
            good.append(r)
    dropped = len(rows) - len(good)
    if dropped:
        print(f"  {path}: dropped {dropped} rows (missing audio, empty, > {MAX_DUR} s, or a test/dev sentence)", flush=True)
    return good


def source_note(path, use):
    side = pathlib.Path(path).with_suffix(".source.json")
    if side.exists():
        return dict(json.loads(side.read_text(encoding="utf-8")), use=use)
    print(f"  no {side.name}: the model card will ask you to fill in this dataset's license", flush=True)
    return {"name": pathlib.Path(path).stem, "url": "", "license": "UNKNOWN - fill in before publishing", "use": use}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, help="language code (languages/<code>.json) or a config path")
    ap.add_argument("--data", help="output folder (default work/<code>/data)")
    ap.add_argument("--cv", help="extracted Common Voice locale folder")
    ap.add_argument("--fleurs", action="store_true", help="also use FLEURS (if the config names a FLEURS config)")
    ap.add_argument("--extra", action="append", default=[], help="extra training manifest (repeatable)")
    ap.add_argument("--extra-test", action="append", default=[], help="extra test manifest (repeatable)")
    ap.add_argument("--verify", action="append", default=[], help="extra training manifest to check with the teacher first")
    ap.add_argument("--verify-other", action="store_true", help="also use CV other.tsv clips the teacher confirms")
    ap.add_argument("--verify-cer", type=float, default=0.15, help="max loose CER for --verify (other.tsv uses 0.10)")
    ap.add_argument("--teacher", help="Whisper model for label checks (default: the language config's teacher)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="max clips per source/split (smoke tests)")
    a = ap.parse_args()

    lang = load_lang(a.lang)
    out = pathlib.Path(a.data or f"work/{lang.code}/data"); out.mkdir(parents=True, exist_ok=True)
    sources, held, other = [], set(), []
    if a.cv:
        other, held, note = common_voice(pathlib.Path(a.cv), lang, out, a.limit)
        sources.append(note)
    if a.fleurs:
        if not lang.fleurs:
            print("the language config has no FLEURS config ('fleurs'); skipping FLEURS", flush=True)
        else:
            note = fleurs(lang, out, a.limit)
            if note:
                sources.append(note)
                held |= {lang.norm(r["text"]) for r in read_jsonl(out / "eval_fleurs_test.jsonl")}
    for f in a.extra_test:
        rows = load_extra(f, lang, set(), a.limit)
        write_jsonl(out / f"eval_{pathlib.Path(f).stem}_test.jsonl", rows)
        held |= {lang.norm(r["text"]) for r in rows}
        sources.append(source_note(f, "Evaluation only"))

    teacher = None
    if a.verify or a.verify_other:
        name = a.teacher or lang.teacher
        if not name:
            raise SystemExit("label checks need --teacher (or 'teacher' in the language config)")
        teacher = Teacher(name, lang, pick_device(a.device))
    if a.verify_other and other:
        print(f"checking {len(other)} unvalidated Common Voice clips with the teacher...", flush=True)
        write_jsonl(out / "train_cv_other.jsonl", verify(other, teacher, lang, 0.10))
    for f in a.extra + a.verify:
        rows = load_extra(f, lang, held, a.limit)
        if f in a.verify:
            print(f"checking {f} with the teacher...", flush=True)
            rows = verify(rows, teacher, lang, a.verify_cer)
        write_jsonl(out / f"train_{pathlib.Path(f).stem}.jsonl", rows)
        sources.append(source_note(f, "Training" + (" (labels checked by the teacher)" if f in a.verify else "")))

    if not list(out.glob("train_*.jsonl")):
        sys.exit("no training data: give --cv, --fleurs and/or --extra")
    (out / "sources.json").write_text(json.dumps({"language": lang.code, "sources": sources, "teacher": (a.teacher or lang.teacher) if teacher else None},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out / 'sources.json'}", flush=True)


if __name__ == "__main__":
    random.seed(0)
    main()
