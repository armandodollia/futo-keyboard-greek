"""Write a Hugging Face model card (README.md) and ATTRIBUTION.md for a trained voice model.

Reads what the other steps left behind: <data>/sources.json (datasets and licenses, from prepare_data.py),
<model>/train_info.json (recipe and PyTorch test scores, from train.py) and the evaluate.py --json results (FUTO mode).
Datasets without a known license are flagged at the top of the card: fix them before publishing.

Usage: python model_card.py --lang el --size small --bin work/el/small/Greek-244.bin --model work/el/small/model \
         --results work/el/small/results.json --author "Your Name" --out work/el/small/card
"""
import argparse
import datetime
import hashlib
import json
import pathlib

from common import SIZES, load_lang

TOOLS = [
    ("[OpenAI Whisper](https://github.com/openai/whisper) (`openai/whisper-{size}`)", "Starting weights", "MIT, © 2022 OpenAI"),
    ("[FUTO whisper-acft](https://github.com/futo-org/whisper-acft)", "Audio-context fine-tuning method (`acft.py`)", "MIT, © FUTO"),
    ("[whisper.cpp](https://github.com/ggml-org/whisper.cpp)", "ggml format and quantization", "MIT, © The ggml authors"),
    ("[Hugging Face Transformers](https://github.com/huggingface/transformers)", "Training", "Apache-2.0"),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--size", required=True, choices=SIZES)
    ap.add_argument("--bin", required=True, help="the FUTO .bin")
    ap.add_argument("--model", help="train.py output folder (for train_info.json)")
    ap.add_argument("--results", help="evaluate.py --json file")
    ap.add_argument("--data", help="manifest folder with sources.json (default work/<code>/data)")
    ap.add_argument("--stock", help="file name of FUTO's stock model in the results, for the comparison row")
    ap.add_argument("--author", default="", help="shown in the card; leave empty to fill in later")
    ap.add_argument("--license", default="mit", help="license of the weights (Whisper is MIT; check your data's terms)")
    ap.add_argument("--out", required=True, help="folder for README.md and ATTRIBUTION.md")
    a = ap.parse_args()

    lang = load_lang(a.lang)
    label = SIZES[a.size]
    binp = pathlib.Path(a.bin)
    data = pathlib.Path(a.data or f"work/{lang.code}/data")
    src = json.loads((data / "sources.json").read_text(encoding="utf-8")) if (data / "sources.json").exists() else {"sources": []}
    info = {}
    if a.model and (pathlib.Path(a.model) / "train_info.json").exists():
        info = json.loads((pathlib.Path(a.model) / "train_info.json").read_text(encoding="utf-8"))
    results = json.loads(pathlib.Path(a.results).read_text(encoding="utf-8")) if a.results else {}

    tools = [(n.format(size=a.size), u, l) for n, u, l in TOOLS]
    teachers = {t for t in (info.get("teacher"), src.get("teacher")) if t}
    for t in sorted(teachers):
        use = "Distillation teacher" if t == info.get("teacher") else "Checked training labels"
        if t == info.get("teacher") and t == src.get("teacher"):
            use = "Distillation teacher; checked training labels"
        tools.append((f"`{t}`", use, "see its model card"))
    unknown = [s["name"] for s in src["sources"] if "UNKNOWN" in s.get("license", "")]

    ds_rows = "\n".join(f"| {'[' + s['name'] + '](' + s['url'] + ')' if s.get('url') else s['name']} | {s.get('use', '')} | "
                        f"{s.get('license', '')} |" for s in src["sources"]) or "| (none recorded) | | |"
    citations = "\n".join(f"- {s['citation']}" for s in src["sources"] if s.get("citation"))
    attribution = f"""# Attribution

## Models and code

| Component | Use | License |
|---|---|---|
{chr(10).join(f'| {n} | {u} | {l} |' for n, u, l in tools)}

## Data

| Dataset | Use | License |
|---|---|---|
{ds_rows}

{citations}
- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*, 2022.

No audio is redistributed with this model.
"""

    ours = binp.name
    futo_rows = []
    sets = sorted({s for m in results.values() for s in m})
    for key, title in ((ours, f"**This model** (`{ours}`)"), (a.stock, f"FUTO stock (`{a.stock}`)")):
        if key and key in results:
            cells = " | ".join(f"{results[key][s]['wer']:.1f}%" if s in results[key] else "n/a" for s in sets)
            reps = sum(results[key][s].get("repetitions", 0) for s in results[key])
            n = sum(results[key][s].get("n", 0) for s in results[key])
            futo_rows.append(f"| {title} | {cells} | {reps}/{n} |")
    futo_table = (f"| Model | {' | '.join(s + ' WER' for s in sets)} | Runaway repetitions |\n"
                  f"|---|{'---|' * len(sets)}---|\n" + "\n".join(futo_rows)) if futo_rows else "_Run evaluate.py --json and pass --results._"
    pt = info.get("test_wer_pytorch", {})
    pt_table = "\n".join(f"| {k} | {v:.1f}% |" for k, v in pt.items())
    clips = ", ".join(f"{k}: {v:,}" for k, v in info.get("clips", {}).items())
    kd = info.get("kd")
    warn = (f"> **Before publishing:** the license of {', '.join(unknown)} is unknown. Add a `.source.json` next to the "
            f"manifest (see the toolkit README), rerun prepare_data.py, then regenerate this card.\n\n") if unknown else ""

    card = f"""---
language:
- {lang.code}
license: {a.license}
base_model: openai/whisper-{a.size}
pipeline_tag: automatic-speech-recognition
library_name: transformers
tags:
- whisper
- whisper.cpp
- futo-keyboard
- speech-recognition
metrics:
- wer
---

{warn}# Whisper {a.size} {lang.name} for FUTO Keyboard (`{lang.name}-{label}`)

Speech recognition for {lang.name} in [FUTO Keyboard](https://keyboard.futo.org) voice input (stock FUTO or the
FUTO Keyboard Polyglot build), fine-tuned from OpenAI `whisper-{a.size}` and ACFT-tuned so it works with FUTO's short,
dynamic audio windows.{f' By {a.author}.' if a.author else ''}

## Files

| File | What it is |
|---|---|
| `{ours}` | **For FUTO Keyboard.** whisper.cpp ggml, ACFT-tuned. Also works in any whisper.cpp app. SHA-256 `{sha256(binp) if binp.exists() else 'n/a'}` |

**Install:** copy the `.bin` to the phone, then in FUTO Keyboard: Settings → Languages & Models → {lang.name} →
Voice Input → Import (or Replace), and pick `{ours}`.

## Results

**FUTO mode:** whisper.cpp with the audio context cut to the clip length plus 64 frames, as FUTO Keyboard runs it;
greedy decoding, language forced to {lang.name}. Same clips for every model. WER, lower is better.

{futo_table}

**Full 30 s window, before ACFT** (Transformers, whole test sets):

| Test set | WER |
|---|---|
{pt_table or '| n/a | |'}

## Training

- **Starting point:** `{info.get('base', f'openai/whisper-{a.size}')}`
- **Data:** {info.get('hours', '?')} h ({clips or 'see ATTRIBUTION.md'}). Test and dev sentences were kept out of all training data, and Common Voice test/dev speakers too.
- **Recipe:** {info.get('epochs', '?')} epochs = {info.get('steps', '?')} steps, batch {info.get('batch', '?')} × {info.get('accum', 1)};
  AdamW, learning rate {info.get('lr', '?')}, {info.get('warmup', '?')} warmup steps, linear decay to 2%, weight decay 0.01,
  gradient clipping 1.0; SpecAugment masks {info.get('mask', '?')}{'; bf16 autocast' if info.get('device') == 'cuda' else ''}.
- **Distillation:** {f"teacher `{info['teacher']}`, T = {kd['T']}, loss = {kd['ce']} × CE + {kd['kl']} × KL" if kd else 'none (hard labels only)'}.
- **Checkpoint selection:** mean WER on the dev sets ({', '.join(f'{k}: {v} clips' for k, v in info.get('dev_sets', {}).items()) or 'n/a'});
  stock {info.get('stock_dev_wer', '?')}% → best {info.get('best_dev_wer', '?')}%.
- **ACFT:** following [FUTO's method](https://github.com/futo-org/whisper-acft): the encoder sees only the clip's frames
  and learns to match the decoder states of a frozen full-window copy (Adam, lr 1e-6, MSE).
- **Conversion:** ggml via the toolkit's `convert_ggml.py`, quantized with whisper.cpp `whisper-quantize`.
- Built with the FUTO Keyboard Polyglot voice training toolkit (`training/voice`), {datetime.date.today():%Y-%m}.

## Limitations

- Trained mostly on read speech (Common Voice style); spontaneous, noisy or dialectal speech may do worse.
- Punctuation and casing follow the training transcripts and can be inconsistent.
- tiny and base are clearly less accurate than small; use small if the phone can run it.

## License and attribution

Weights: `{a.license}`. Training data and tools are credited in `ATTRIBUTION.md`; keep it with any redistribution
(CC-BY datasets require it).
"""
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(card, encoding="utf-8")
    (out / "ATTRIBUTION.md").write_text(attribution, encoding="utf-8")
    print(f"wrote {out / 'README.md'} and {out / 'ATTRIBUTION.md'}" + (f"; UNKNOWN licenses: {unknown}" if unknown else ""))


if __name__ == "__main__":
    main()
