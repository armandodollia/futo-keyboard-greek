"""Score models on the language's test sets: WER, CER, loose CER and runaway repetitions.

Each model is either
  - a whisper.cpp .bin -> "FUTO mode": whisper-cli with the audio context cut to the clip's length plus 64 frames
    (-ac), greedy decoding, language forced. This is how FUTO Keyboard runs a voice model, and the number to trust
    when deciding what to ship. Compare against FUTO's stock model of the same size by passing its .bin too.
  - a Hugging Face folder or id -> Transformers with the full 30 s window (sanity checks; not what the phone does).

Sets: every <data>/eval_*_test.jsonl (or --sets), --n clips each (a fixed random sample, the same for every model).
--json merges the results into a file that model_card.py reads.

Usage: python evaluate.py --lang el --models out/Greek-244.bin stock/futo-multilingual-244.bin --whisper-cpp ../whisper.cpp
"""
import argparse
import json
import math
import os
import pathlib
import random
import subprocess
import tempfile

import jiwer

from common import SR, find_tool, load_audio, load_lang, pick_device, read_jsonl, repeated, write_wav


def sample(path, n):
    rows = [r for r in read_jsonl(path) if r.get("dur", 0) <= 30]
    random.Random(7).shuffle(rows)
    return rows[:n] if n else rows


def run_ggml(model, clips, lang, cli, threads):
    hyps = []
    for wav, dur in clips:
        ac = min(1500, int(math.ceil(dur * 50)) + 64)       # FUTO-style: the clip's frames plus a little margin
        o = subprocess.run([cli, "-m", str(model), "-f", str(wav), "-l", lang.code, "-ac", str(ac), "-nt", "-np",
                            "-t", str(threads), "-bs", "1", "-bo", "1"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if o.returncode:
            raise SystemExit(f"whisper-cli failed on {wav}:\n{o.stderr[-800:]}")
        hyps.append(o.stdout.strip())
    return hyps


def run_hf(model_id, rows, lang, device):
    import torch
    from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(model_id)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device).eval()
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_id)
    except OSError:
        model.generation_config = GenerationConfig.from_pretrained("openai/whisper-small")
    hyps = []
    for i in range(0, len(rows), 16):
        audio = [load_audio(r["audio"])[: SR * 30] for r in rows[i:i + 16]]
        feats = proc.feature_extractor(audio, sampling_rate=SR, return_tensors="pt").input_features.to(device, dtype)
        with torch.no_grad():
            ids = model.generate(feats, language=lang.whisper_language, task="transcribe", max_new_tokens=220)
        hyps += [h.strip() for h in proc.batch_decode(ids, skip_special_tokens=True)]
    return hyps


def score(refs, hyps, lang):
    r = [lang.norm(t) or "-" for t in refs]
    h = [lang.norm(t) or "-" for t in hyps]
    return {"wer": round(jiwer.wer(r, h) * 100, 2), "cer": round(jiwer.cer(r, h) * 100, 2),
            "loose_cer": round(jiwer.cer([lang.loose(t) or "-" for t in refs], [lang.loose(t) or "-" for t in hyps]) * 100, 2),
            "repetitions": sum(repeated(x) for x in hyps), "n": len(refs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--models", nargs="+", required=True, help=".bin files (FUTO mode) and/or HF folders/ids")
    ap.add_argument("--data", help="manifest folder (default work/<code>/data)")
    ap.add_argument("--sets", nargs="*", help="test manifests (default: <data>/eval_*_test.jsonl)")
    ap.add_argument("--n", type=int, default=200, help="clips per set (0 = all)")
    ap.add_argument("--whisper-cpp", default="", help="whisper.cpp checkout or bin folder (for whisper-cli)")
    ap.add_argument("--threads", type=int, default=max(1, min(8, (os.cpu_count() or 4))))
    ap.add_argument("--device", default="auto", help="for Hugging Face models")
    ap.add_argument("--json", help="merge results into this file")
    ap.add_argument("--show", type=int, default=2, help="print this many example transcripts per set")
    a = ap.parse_args()

    lang = load_lang(a.lang)
    data = pathlib.Path(a.data or f"work/{lang.code}/data")
    sets = {pathlib.Path(p).stem.removeprefix("eval_").removesuffix("_test"): sample(p, a.n)
            for p in (a.sets or sorted(data.glob("eval_*_test.jsonl")))}
    if not sets:
        raise SystemExit(f"no test sets (looked for {data}/eval_*_test.jsonl)")
    cli = find_tool("whisper-cli", a.whisper_cpp) if any(m.endswith(".bin") for m in a.models) else None
    results = json.loads(pathlib.Path(a.json).read_text(encoding="utf-8")) if a.json and pathlib.Path(a.json).exists() else {}

    with tempfile.TemporaryDirectory() as tmp:
        wavs = {}          # whisper-cli wants 16 kHz wav; convert each clip once and reuse it for every model
        if cli:
            for name, rows in sets.items():
                wavs[name] = []
                for i, r in enumerate(rows):
                    p = pathlib.Path(tmp) / f"{name}_{i}.wav"
                    audio = load_audio(r["audio"])
                    write_wav(p, audio)
                    wavs[name].append((p, len(audio) / SR))
        for m in a.models:
            key = pathlib.Path(m).name
            mode = "futo" if m.endswith(".bin") else "hf"
            for name, rows in sets.items():
                hyps = run_ggml(m, wavs[name], lang, cli, a.threads) if mode == "futo" else \
                    run_hf(m, rows, lang, pick_device(a.device))
                s = score([r["text"] for r in rows], hyps, lang)
                results.setdefault(key, {})[name] = dict(s, mode=mode)
                print(f"{key:36s} {name:12s} {'FUTO mode' if mode == 'futo' else 'HF 30 s  '}  WER {s['wer']:5.1f}%  "
                      f"CER {s['cer']:5.1f}%  loose CER {s['loose_cer']:5.1f}%  repetitions {s['repetitions']}/{s['n']}", flush=True)
                for r, h in list(zip(rows, hyps))[: a.show]:
                    print(f"    ref: {r['text'][:110]}\n    hyp: {h[:110]}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
