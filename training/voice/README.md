# Voice input models for FUTO Keyboard

Train a Whisper voice-input model for your language and load it into FUTO Keyboard, either the stock app or
FUTO Keyboard Polyglot. The result is a single whisper.cpp `.bin` file.

The recipe is the one behind the Greek and Albanian models. Measured in FUTO mode (see [Evaluate](#6-evaluate)), word
error rate (WER) on the Common Voice test set, lower is better:

| Model | Greek | Albanian |
|---|---|---|
| FUTO stock Multilingual-244 | 31.5% | 94.1% |
| Fine-tuned whisper-small, this recipe | **14.7%** | **33.4%** |

Greek had about 100 h of training audio. Albanian had 9 h, so a little data already helps a lot.

```
prepare_data.py -> train.py -> acft.py -> convert_ggml.py -> evaluate.py -> model_card.py
 manifests         fine-tune    FUTO        whisper.cpp       FUTO-mode      README +
 (jsonl)           (+ KD)       ACFT        .bin, q8_0        WER/CER        ATTRIBUTION
```

`pipeline.py` runs every step with one command.

## What you need

- **Your language must be one Whisper knows.** FUTO and whisper.cpp force the language by Whisper's code: `el`, `sq`,
  `cy`, `sw` and so on, [99 in total](https://github.com/openai/whisper#available-models-and-languages). A language
  Whisper doesn't know would need a new language token, which this toolkit doesn't do.
- **Python 3.10+ and an NVIDIA GPU.** A GPU with 24 GB trains `small` without gradient checkpointing (`--no-ckpt`, peak
  17.4 GB at batch 16). Gradient checkpointing (the default) needs much less. `train.py --probe` measures one
  worst-case step, and `--batch 8 --accum 2` halves memory at the same effective batch. CPU works for smoke tests only.
- **whisper.cpp** for quantizing and for the FUTO-mode test. The ggml conversion itself is pure Python.
  ```sh
  git clone https://github.com/ggml-org/whisper.cpp
  cmake -S whisper.cpp -B whisper.cpp/build -DCMAKE_BUILD_TYPE=Release
  cmake --build whisper.cpp/build --config Release -j
  ```
  Then pass `--whisper-cpp path/to/whisper.cpp`, or set `WHISPER_CPP`. The scripts look in `build/bin` and
  `build/bin/Release`.
- **Packages:** install PyTorch for your platform first (<https://pytorch.org/get-started/locally/>), then run
  `pip install -r requirements.txt`.

### Time on one RTX 4090 (from the training logs)

| Run | Data | Training | ACFT |
|---|---|---|---|
| Greek small, 4 epochs, batch 16, no checkpointing | 100 h, 67k clips, 16,708 steps | 2 h 05 min | 24 min |
| Greek base, same data, small as KD teacher | same | 1 h 23 min | 16 min |
| Greek tiny, same data, small as KD teacher | same | 1 h 16 min | 12 min |
| Albanian small, 10 epochs | 7.8 h, 2,050 steps | 19 min | 25 min |

The whole small pipeline takes 2 to 3 hours for 100 h of audio, and under an hour for a 10 h language. Checkpointing
adds 25 to 35% to training time. A 200-clip FUTO-mode test takes about 2 to 4 minutes per model on the CPU. Slower
GPUs take proportionally longer.

## Step by step for a new language

The examples use Welsh (`cy`). Swap in your own code.

### 1. Describe the language (optional)

Copy `languages/template.json` to `languages/cy.json`. Without a file, `--lang cy` still works with default
normalisation. See [Language config](#language-config).

### 2. Get Common Voice

Download your language from the [Mozilla Data Collective](https://datacollective.mozillafoundation.org) (Common Voice
moved there; you need a free account and must accept the terms). Extract it; you need the folder that holds
`validated.tsv`, `test.tsv`, `dev.tsv` and `clips/`, e.g. `cv-corpus-27.0-2026-09-11/cy`.

Keep datasets outside this repository. `work/` is git-ignored, but raw data shouldn't go here either.

### 3. Run the pipeline

```sh
python pipeline.py --lang cy --size small --cv /data/cv-corpus-27.0-2026-09-11/cy --fleurs \
    --whisper-cpp /src/whisper.cpp --author "Your Name"
```

This writes `work/cy/small/Welsh-244.bin`. The name follows FUTO's convention: 244, 74 or 39 M parameters for small,
base or tiny. Everything the pipeline does is logged in `work/cy/small/pipeline.log`. A rerun skips steps whose output
exists, so an interrupted run resumes; `--steps train,acft --force` redoes selected steps. `--dry-run` prints the
commands without running them.

Useful options:

| Option | Default | Notes |
|---|---|---|
| `--size` | small | small is clearly the most accurate; train base/tiny for slow phones |
| `--epochs` | 10 below 15 h of data, 6 below 50 h, else 4 | Greek, 100 h, peaked around 4 epochs; Albanian, 8 h, was still improving slowly at 10 |
| `--lr` | small 1.25e-5, base 2.5e-5, tiny 3.75e-5 | |
| `--no-ckpt` | off | faster if the GPU has room; `train.py --probe` prints the peak memory of one worst-case step |
| `--teacher` | none | knowledge distillation, see below |
| `--stock` | none | FUTO's stock `.bin` of the same size, so the results table shows whether yours is better |
| `--extra`, `--extra-test`, `--verify` | | your own data, see below |

### 4. Train the smaller sizes (optional)

Once small is done, train base and tiny with it as the distillation teacher. This is what the Greek base and tiny
models used:

```sh
python pipeline.py --lang cy --size base --teacher work/cy/small/model --whisper-cpp /src/whisper.cpp
python pipeline.py --lang cy --size tiny --teacher work/cy/small/model --whisper-cpp /src/whisper.cpp
```

The data from step 3 is reused (`work/cy/data`).

### 5. Install on the phone

Copy the `.bin` to the phone over USB, a cloud drive, Syncthing or similar. Then, in FUTO Keyboard (or Polyglot):

**Settings → Languages & Models → your language → Voice Input → Import** (it says **Replace** if a model is already
set), and pick the file.

You can also open the `.bin` from a file manager with FUTO Keyboard, which offers to import it as a *Voice Input
Model*. The language has to be added to the keyboard first (Languages & Models → Add language).

### 6. Evaluate

`evaluate.py` runs the `.bin` in **FUTO mode**, the way the keyboard runs it:

- whisper.cpp with the audio context cut to the clip length plus 64 frames (`-ac`)
- greedy decoding
- the language forced

It reports WER, CER, "loose" CER (forgiving the diacritics in your config) and *runaway repetitions* (looping output,
which is what ACFT prevents). **Decide what to ship on this number, not on the Transformers scores in the training log.**
Those use the full 30 s window, which the phone never does.

```sh
python evaluate.py --lang cy --models work/cy/small/Welsh-244.bin old/Welsh-244-v1.bin --whisper-cpp /src/whisper.cpp
```

A `.bin` is scored in FUTO mode. A Hugging Face folder or id (e.g. `openai/whisper-small`) is scored with Transformers.

### 7. Publish (optional)

`work/cy/small/card/` holds a Hugging Face model card (`README.md`) and `ATTRIBUTION.md`, built from the recorded data
sources, recipe and scores. Read it, fill in anything marked UNKNOWN, and upload it together with the `.bin`. Uploading
the Transformers folders (`model/`, `model-acft/`) as well lets others fine-tune further.

## Your own data

### Manifest format

A manifest is a `.jsonl` file with one clip per line:

```json
{"audio": "clips/0001.wav", "text": "Bore da, sut wyt ti?", "dur": 2.4, "speaker": "s12", "group": "book-3"}
```

| Field | Required | Meaning |
|---|---|---|
| `audio` | yes | Any format ffmpeg reads. A relative path is resolved against the manifest's folder |
| `text` | yes | The transcript as it should be typed: normal casing and punctuation. The model learns this style |
| `dur` | no | Seconds; computed if missing. Clips over 29.5 s are dropped because Whisper's window is 30 s |
| `src` | no | Label in the logs (default: the file name) |
| `group` | no | With `train.py --cap-hours-per-group H`, at most H hours per group, so one audiobook or narrator can't dominate |

Pass the manifest to the pipeline or to `prepare_data.py`:

- `--extra F`: training data. Rows are checked, the language's `clean` rules are applied, and sentences that also
  appear in any test/dev set are dropped.
- `--extra-test F`: a held-out test set, e.g. your own recordings or a spontaneous-speech corpus. It's evaluated,
  never trained on.
- `--verify F`: training data with labels you don't fully trust, such as YouTube subtitles or TTS clips. A large
  teacher transcribes each clip, and only clips whose label it roughly reproduces are kept (loose CER ≤ `--verify-cer`,
  default 0.15). This needs `--label-teacher` or `teacher` in the language config, and a GPU in practice.
- `--verify-other`: the same check for Common Voice's `other.tsv`, clips recorded but not yet voted on (threshold
  0.10). It can add a lot of data for small languages.

Put a sidecar file next to each manifest, named `F.source.json` (for `mydata.jsonl`, that is `mydata.source.json`), so
the model card can credit it:

```json
{"name": "My Corpus v2", "url": "https://example.org/corpus", "license": "CC-BY-4.0",
 "citation": "Author et al., Title, 2025"}
```

Without it, the card flags the dataset's license as UNKNOWN.

### Data licensing and attribution

- **Common Voice is CC0.** Credit is appreciated but not required. Don't try to identify speakers.
- **FLEURS is CC-BY-4.0.** Credit is required; the generated `ATTRIBUTION.md` does it.
- **Everything else varies.** YODAS (Creative Commons YouTube) is CC-BY-3.0 and credits the uploaders. LibriVox is
  public domain. Many "open" corpora are non-commercial (CC-BY-NC) or research-only: a model trained on them may not be
  shareable, or only under the same terms. Check before you publish.
- **Don't scrape arbitrary YouTube or podcasts.** Terms of service and copyright apply. Use datasets that publish a
  license.
- **The weights start from OpenAI Whisper (MIT).** The toolkit defaults the card to MIT (`--license`); pick what your
  data allows.
- **Teacher-made labels** carry the teacher model's license terms too. Most Whisper fine-tunes are MIT or Apache-2.0,
  but check.

## Language config

`languages/<code>.json` (see `el.json`, `sq.json`, `template.json`):

| Field | Meaning |
|---|---|
| `code` | Whisper / whisper.cpp language code (`el`, `sq`, `cy`, …) |
| `name` | English name; used in file names (`Greek-244.bin`) and the card |
| `whisper_language` | Whisper's language name (`greek`); derived from `code` if null |
| `fleurs` | FLEURS config name (`el_gr`), or null if FLEURS lacks the language |
| `teacher` | Default large model for `--verify` / `--verify-other` label checks (e.g. `openai/whisper-large-v3`, or a community large-v3 fine-tune for your language) |
| `normalize` | How transcripts are compared for WER/CER: `unicode` form, `lowercase`, character `replace` map (Greek folds final ς into σ), `strip_marks` (drop all accents) |
| `loose` | Extra forgiveness for "loose CER" and teacher checks: `replace` map (Albanian ë→e, ç→c, which subtitles often drop), `strip_marks` (Greek tonos) |
| `clean` | `[regex, replacement]` pairs applied to *training* transcripts, e.g. unify apostrophes. Use sparingly: the model learns to write what it's trained on |

Punctuation is always ignored when scoring. Casing is ignored when `lowercase` is true.

## Knowledge distillation (optional)

With `--teacher`, the student learns from the teacher's full token probabilities as well as the transcript. The loss is
0.8 × cross-entropy + 1.0 × KL at temperature 2, the Distil-Whisper recipe. Two cases have worked:

- **Base or tiny from your own fine-tuned small** (`--teacher work/<code>/small/model`). This gave the Greek base and
  tiny models.
- **Small from a strong large model for your language.** Albanian small used a community large-v3-turbo fine-tune:
  CV test 40.5% without it, 32.3% with it (plus 1.3 h more data).

The teacher can come from the large-v3 family. That family has 128 mel bins instead of 80 and one extra vocabulary token
(`<|yue|>`), plus a renamed `<|nocaptions|>`. `train.py` gives the teacher its own features and maps tokens by
string. The cost is one teacher forward pass per step and the teacher's memory, about 1.6 GB for large-v3-turbo in
bf16. Without a good teacher, skip KD: plain fine-tuning is the default and works.

## ACFT: why it's needed

FUTO doesn't pad dictations to 30 s. It runs the Whisper encoder only on the frames the audio needs, which is much
faster on a phone. A model fine-tuned the normal way has only seen 30 s windows, and with a short window it tends to
loop ("και και και …") or stop early.

`acft.py` fixes this with [FUTO's audio-context fine-tuning](https://github.com/futo-org/whisper-acft). The model sees
shortened windows and learns to match a frozen copy of itself that sees the full window. It takes a few minutes to
half an hour, and it is not optional for FUTO. The ACFT loss is much larger for tiny and base than for small (their
last decoder layer has large activations), and that is expected.

## Files

| File | What it does |
|---|---|
| `pipeline.py` | Runs every step below, resumable, logged |
| `prepare_data.py` | Builds the Common Voice, FLEURS and own-data manifests; teacher checks; `sources.json` for attribution |
| `train.py` | Fine-tuning with optional KD; checkpoint picked on dev WER; `--probe` for memory |
| `acft.py` | FUTO audio-context fine-tuning |
| `convert_ggml.py` | Hugging Face → whisper.cpp ggml (byte-identical to whisper.cpp's converter apart from float rounding in the mel filters), then `whisper-quantize` (q8_0 default) |
| `evaluate.py` | FUTO-mode (or Transformers) WER, CER, loose CER and repetitions; `--json` for the card |
| `model_card.py` | Hugging Face `README.md` + `ATTRIBUTION.md` |
| `common.py` | Language config, normalisation, manifests, audio I/O |
| `languages/` | Language configs: Greek, Albanian and a template |

## Tips and caveats

- **Build the test set before you train, and keep it clean.** Common Voice test/dev speakers and sentences are removed
  from training automatically, because CV repeats sentences across speakers and leaking them inflates scores.
- **Ship only if FUTO mode beats the previous model.** Compare against FUTO's stock model (`--stock`) and against your
  last release.
- **q8_0 is the tested quantization.** It is half the size of f16 (small: 252 vs 465 MB) with no measurable accuracy
  loss. Smaller quantizations (`--quantize q5_1`) are untested here.
- **FLEURS needs `datasets<4`** (it's a loading-script dataset). With a newer `datasets`, the prepare step skips it
  with a message.
- **Long audio isn't handled.** Audiobooks or podcasts without clip-level transcripts must first be cut into ≤ 30 s
  clips with text. The Greek model did this with a large Whisper as labeller plus quality filters, which is not part of
  this toolkit. `--verify` covers the case where you already have clips and rough labels.
- **Dialects are hard.** Albanian Gheg spontaneous speech stayed at 65% WER after training, versus 33% for standard
  read speech. Add real dialect data if it matters.
