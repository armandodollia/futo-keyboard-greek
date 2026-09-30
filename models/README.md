# Models for FUTO Keyboard Polyglot

Ready-made Greek and Albanian models for [FUTO Keyboard Polyglot](../README.md): voice input, typing (next-word
prediction and autocorrect), dictionaries and dictation cleanup. They were built with the tooling in
[`training/`](../training/README.md).

The files are attached to the GitHub prerelease
[`models-v1`](https://github.com/armandodollia/futo-keyboard-polyglot/releases/tag/models-v1). This folder has only the
catalogue, the licences and the credits.

**Licensing in one paragraph.** FUTO Keyboard's licence (`LICENSE-FUTO.md`) covers the app and the patches. It does
**not** cover these models: they are separate works with their own licences, listed per file below. They are made for
the FUTO app, but most of them also work with any whisper.cpp or llama.cpp program. Credits for every base model,
teacher and dataset are in [`ATTRIBUTION.md`](ATTRIBUTION.md). Required notices, including the Gemma notice for the
cleanup model, are in [`NOTICE`](NOTICE). Licence texts are in [`LICENSES/`](LICENSES). If you redistribute a file,
keep `NOTICE`, `ATTRIBUTION.md` and the matching licence with it.

## Files

Base URL: `https://github.com/armandodollia/futo-keyboard-polyglot/releases/download/models-v1/`

| File | What it is | Size | Licence |
|---|---|---|---|
| [`Greek-244-v4.bin`](https://github.com/armandodollia/futo-keyboard-polyglot/releases/download/models-v1/Greek-244-v4.bin) | Voice: Whisper small (244M), Greek, ACFT, whisper.cpp q8_0 | 252 MB | MIT |
| [`Greek-74-v4.bin`](https://github.com/armandodollia/futo-keyboard-polyglot/releases/download/models-v1/Greek-74-v4.bin) | Voice: Whisper base (74M), Greek, ACFT, whisper.cpp q8_0 | 78 MB | MIT |
| [`Greek-39-v4.bin`](https://github.com/armandodollia/futo-keyboard-polyglot/releases/download/models-v1/Greek-39-v4.bin) | Voice: Whisper tiny (39M), Greek, ACFT, whisper.cpp q8_0 | 42 MB | MIT |
| [`Albanian-244-v2.bin`](https://github.com/armandodollia/futo-keyboard-polyglot/releases/download/models-v1/Albanian-244-v2.bin) | Voice: Whisper small (244M), Albanian incl. Gheg, ACFT, whisper.cpp q8_0 | 252 MB | MIT (see [data caveat](#licences-and-data-caveats)) |
| [`el_typing_v1.1_Q8_0.gguf`](https://github.com/armandodollia/futo-keyboard-polyglot/releases/download/models-v1/el_typing_v1.1_Q8_0.gguf) | Typing: Greek transformer LM (35.5M Llama, 16k SentencePiece), Q8_0 | 45 MB | MIT |
| [`sq_typing_v1.1_Q8_0.gguf`](https://github.com/armandodollia/futo-keyboard-polyglot/releases/download/models-v1/sq_typing_v1.1_Q8_0.gguf) | Typing: Albanian transformer LM (35.5M Llama, 16k SentencePiece), Q8_0 | 45 MB | MIT |
| [`Greek-main_el.dict`](https://github.com/armandodollia/futo-keyboard-polyglot/releases/download/models-v1/Greek-main_el.dict) | Dictionary: Greek, 908,211 words, 608 flagged offensive | 8.9 MB | GPL-3.0 |
| [`Greek-main_el.combined`](https://github.com/armandodollia/futo-keyboard-polyglot/releases/download/models-v1/Greek-main_el.combined) | Source wordlist of `Greek-main_el.dict` (AOSP `.combined` text) | 31 MB | GPL-3.0 |
| [`Albanian-main_sq.dict`](https://github.com/armandodollia/futo-keyboard-polyglot/releases/download/models-v1/Albanian-main_sq.dict) | Dictionary: Albanian incl. Gheg, 222,977 words, 107 flagged offensive | 1.1 MB | MIT |
| [`Albanian-main_sq.combined`](https://github.com/armandodollia/futo-keyboard-polyglot/releases/download/models-v1/Albanian-main_sq.combined) | Source wordlist of `Albanian-main_sq.dict` | 4.8 MB | MIT |
| [`Cleanup-v2-q4_0.gguf`](https://github.com/armandodollia/futo-keyboard-polyglot/releases/download/models-v1/Cleanup-v2-q4_0.gguf) | Dictation cleanup: Qwen3-1.7B + merged LoRA, Greek/Albanian/English, Q4_0 | 1.0 GB | Apache-2.0, plus Gemma Terms of Use ([NOTICE](NOTICE)) |

Sizes are in MiB. `SHA256SUMS` in this folder has the same checksums in `sha256sum -c` format.

| File | Bytes | SHA-256 |
|---|---|---|
| `Greek-244-v4.bin` | 264,464,624 | `415e7da71f8ebea62a319beeaacfde75c2285f8b52d28cc3652de7cb475b1d31` |
| `Greek-74-v4.bin` | 81,768,602 | `3507a83e5221ae2ae33143f4c24b2ced4d6428981f43aff95e069e62a5b9459b` |
| `Greek-39-v4.bin` | 43,537,450 | `9861d703fec4dcea55703ddbba5a8523d327c26b665ae32b603c21eb6a152f86` |
| `Albanian-244-v2.bin` | 264,464,624 | `2011e011dffb0efb58799bfaf30fbada789063560e4d661db9e9e8760da03564` |
| `el_typing_v1.1_Q8_0.gguf` | 47,209,440 | `5b4ccc0fe21a5f920b8198d04222cb21b435b56b9547fd8a716aca0d8b5d9ce5` |
| `sq_typing_v1.1_Q8_0.gguf` | 47,044,256 | `1a9ad13033c1a4cb7e9ae71d65865267621bd3932c16a6028cb9088d49947abe` |
| `Greek-main_el.dict` | 9,297,026 | `19f4f8ab74a3a19713d4d650b541cec11767ad4a4ae88baf75cb6514e3e9626e` |
| `Greek-main_el.combined` | 31,992,712 | `360525fc32330d74690a373560af39831800759810d2b2551068bf967e0c718b` |
| `Albanian-main_sq.dict` | 1,170,549 | `3674818631bc1439044ccd99e16eff41120a129e24795d36a1b6786047b7f9f6` |
| `Albanian-main_sq.combined` | 5,011,815 | `002c39d89719bb19841cb526191992ffc9ed844d76b747b60a322d3977a779c7` |
| `Cleanup-v2-q4_0.gguf` | 1,054,422,816 | `3ebd17023804d8de4249a0983a8b3db4ddc49b30a780ec832e6d0d8443676369` |

Check a download with `sha256sum -c SHA256SUMS --ignore-missing` (Linux/macOS) or `certutil -hashfile <file> SHA256`
(Windows).

## Installing

**The easy way:** FUTO Keyboard Polyglot downloads them for you (patch 0004). Open Settings → Languages & Models:
each language card lists what is available for it (for Greek voice, pick small, base or tiny), and Settings → Voice
input → Dictation cleanup has *Download on-device model*. Each file's SHA-256 is checked before it is installed. The
rest of this section is the manual way.

All of these are imported in FUTO Keyboard Polyglot (they also work in stock FUTO Keyboard, except the typing
models, which need the patched build, and the cleanup model, which needs the Polyglot dictation-cleanup feature).
Copy the file to the phone first.

### Voice (`.bin`)

Settings → Languages & Models → Voice Input → import the `.bin` and assign it to Greek or Albanian. Pick the size
for your phone: 244 is the most accurate and needs a recent phone, 74 suits mid-range phones, 39 slow or old ones.
The files are standard whisper.cpp ggml models, so any whisper.cpp app can use them too.

### Typing (`.gguf`)

1. Settings → Developer → turn on **"Allow transformer models on non QWERTY layouts"**. Without it FUTO does not use
   a transformer model on the Greek layout or on Albanian QWERTZ.
2. Settings → Languages & Models → open the language → import the `.gguf` as its transformer (typing) model. You can
   also open the file from a file manager and choose FUTO Keyboard Polyglot.
3. Type in the keyboard layout the model was trained for: FUTO's Greek layout, or FUTO's Albanian layout (QWERTZ
   with its own ë and ç keys).

The models need the Unicode keystroke patch (patch 0001). Stock FUTO passes only a–z keystrokes to the model.

### Dictionary (`.dict`)

Open the `.dict` from a file manager or share it to FUTO Keyboard Polyglot, and import it as the dictionary for its
language. Alternatively, go to Settings → Languages & Models, open the language, and import the dictionary there. The
keyboard language must match the dictionary's locale (`el` or `sq`).

Vulgar swear words are flagged `possibly_offensive`, so FUTO's **Block offensive words** setting (text prediction
settings) decides whether they are suggested. They stay valid words either way, so they are never autocorrected
away. Mild insults and everyday slang are not flagged.

### Dictation cleanup (`.gguf`)

- **On the phone:** Settings → Voice input → Dictation cleanup → **Import on-device model (.gguf)**, then choose the
  local or a parallel strategy. It needs a 64-bit phone with enough free memory for a 1 GB model.
- **On your own server:** serve it with llama-server, for example
  `llama-server -m Cleanup-v2-q4_0.gguf --jinja --chat-template-kwargs '{"enable_thinking":false}' --port 8080`.
  Anything that talks to it must use the prompt format the model was trained on:

  - Qwen3 chat template (ChatML), **thinking disabled**. Raw prompt:
    ```
    <|im_start|>system
    {SYSTEM}<|im_end|>
    <|im_start|>user
    {raw transcript}<|im_end|>
    <|im_start|>assistant
    <think>

    </think>

    ```
  - `{SYSTEM}` is `{task}. Language: {language}. Never translate. Keep slang and swearing exactly. Output only the text.`
  - `{task}` is `Clean up: punctuation, capitals, remove fillers, apply self-corrections, keep the words` (light
    mode) or `Rewrite as a short, clear message` (rambler mode).
  - `{language}` is the language's English name: `Greek`, `Albanian` or `English`.

  Example system prompt: `Clean up: punctuation, capitals, remove fillers, apply self-corrections, keep the words. Language: Greek. Never translate. Keep slang and swearing exactly. Output only the text.`

  The app's on-device path uses exactly this format. Its **remote** path sends a longer, general-purpose system
  prompt written for any chat LLM, which this model was not trained on and has not been evaluated with. To use this
  model remotely with the trained prompt, put a small proxy in front of llama-server that rewrites the system message.

## The download index (`index.json`)

The app reads [`index.json`](index.json) from this folder on `main` (via `raw.githubusercontent.com`) when the
Languages or Dictation cleanup screen is opened, and keeps the last good copy. It makes no other network calls for
this. To publish a model or a new version, edit `index.json`:

```json
{"version": 1, "models": [{
  "id": "voice-el", "kind": "voice", "locale": "el", "title": "Greek Whisper v4",
  "description": "shown before download", "version": 1,
  "size": 264464624, "sha256": "<hex>", "url": "https://...",
  "license": "MIT", "info_url": "https://...",
  "variants": [{"name": "small", "size": 264464624, "sha256": "<hex>", "url": "https://..."}]
}]}
```

- `version` (top level) is the schema version, currently `1`.
- `kind`: `typing` (transformer `.gguf`), `voice` (whisper `.bin`), `dictionary` (`.dict`) or `cleanup` (dictation
  cleanup `.gguf`; `locale` is `*`).
- `locale`: the language code (`el`, `sq`), or a full locale such as `pt_BR` to target one keyboard language only.
- `version` (per model) is an integer. Raise it when you replace the file: phones that downloaded an older version
  show "Update available".
- `size` is in bytes and `sha256` is the file's hex SHA-256. Both are checked; a mismatch deletes the download.
- `url` must be `https`. Redirects (Hugging Face `resolve` URLs go to a CDN) are followed.
- `variants` (optional): alternative files for one slot, e.g. voice model sizes. The top-level `size`, `sha256` and
  `url` are the default and must also appear in the list.
- `license` and `info_url` are shown in the confirmation dialog before the download starts.

## Results

**Voice, FUTO mode.** whisper.cpp with the audio context cut to the clip length, as FUTO Keyboard runs it; greedy,
language forced. WER: lower is better.

| Model | Common Voice test WER | FLEURS test WER | Stock FUTO Multilingual, same size |
|---|---|---|---|
| `Greek-244-v4.bin` | 14.7% | 17.4% | 31.5% / 37.0% |
| `Greek-74-v4.bin` | 18.2% | 26.8% | — |
| `Greek-39-v4.bin` | 23.6% | 33.8% | — |

The Greek small model also scores 10.4% WER on synthetic informal speech (slang, insults, swearing) and 20.8% on
held-out YouTube conversation (🤗 Transformers, before ACFT).

| Model | Common Voice 27 sq test WER | Gheg (Spontaneous Speech) tolerant CER | Stock FUTO Multilingual-244 |
|---|---|---|---|
| `Albanian-244-v2.bin` | 33.4% (tolerant CER 8.7%) | 25.3% (WER 64.6%) | 94.1% WER / 47.3% CER |

"Tolerant CER" ignores case, punctuation, ë/ç differences and a word-final e/a/i/u, so common Gheg/standard spelling differences are not counted as errors. On Gheg the model mostly writes standard Albanian
spelling, which is why its Gheg WER is high even when the meaning is right.

**Typing.** Top-3 accuracy, the way FUTO runs the model. "Autocorrect with typos" means sloppy taps plus one
keystroke error or phonetic misspelling. Neutral text is held-out Common Voice sentences (Greek) or held-out web and
Wikipedia text (Albanian); slang is a held-out set of slang, insult and swearing sentences.

| Model | Next word, neutral | Next word, slang | Autocorrect with typos, neutral | Autocorrect with typos, slang |
|---|---|---|---|---|
| `el_typing_v1.1` | 27.1% | 31.7% | 81.0% | 86.9% |
| stamchry Greek LM (for comparison) | 22.3% | 21.2% | — | — |
| `sq_typing_v1.1` | 41.0% | 32.2% | 87.1% | 80.6% |

On greta44/albanian-error-augmentation (2,495 linguist-made Albanian misspellings), `sq_typing_v1.1` puts the correct
word in its top 3 for 58.2%. The data was not trained on, but its error types informed the model's synthetic typos.

**Dictation cleanup** (`Cleanup-v2`, held-out pairs never trained on):

| Metric | Result |
|---|---|
| Similarity to the target (1 − CER) | 84.5% |
| Swear words kept | 93.1% |
| Output stays in the input language | 99.3% |
| Held-out trailing self-corrections ("… five, no wait, six") | 87.8% |

## Known limitations

- **Offensive-word lists need native review.** The dictionaries' `possibly_offensive` flags come from hand-made stem
  lists (608 Greek, 107 Albanian words). Some vulgar words are surely missing and a few innocent look-alikes may be
  flagged.
- **The Albanian slang material needs native review.** The Albanian informal-sentence list (slang, insults, Gheg
  forms) used for the typing model, dictionary and cleanup data was written with AI help and not yet checked by a
  native speaker. Gheg spelling varies.
- **Albanian voice is limited by data.** It has only about 9 h of training audio, of which 4 h are Gheg. Gheg
  transcripts come out mostly in standard spelling. There is no Albanian tiny or base model.
- **Voice models transcribe swearing verbatim** and do not censor it. The Greek slang test is synthetic (TTS voices).
  Cypriot and other Greek dialects are under-represented.
- **Cleanup, rambler mode, can drop details** (a time, a name, a second request) while shortening, and sometimes
  gets agreement wrong after a corrected word ("ο μικρός … όχι συγγνώμη η μικρή" → "η μικρή είναι άρρωστος"). Albanian
  number corrections are weakest ("në orën pesë jo më fal në gjashtë" → "në gjashtë pesë"). Check the result before
  sending, or use light mode.
- **Cleanup keeps swearing on purpose** (93.1%, not 100%). The app rejects a result that drops swear words and falls
  back to the next option or the raw transcript.
- **Typing models need the patched keyboard** and the Developer setting above. They are trained for FUTO's Greek
  and Albanian layouts only.
- **Typing v1.1 = v1 weights, fixed export.** The first v1 GGUFs (shared privately, never in this catalogue) were
  written without the Q/K RoPE permutation llama.cpp needs, so in the keyboard they agreed with the trained model on
  only ~40% of top predictions. v1.1 is the same checkpoint re-exported with `training/typing-lm/export.py` and
  checked in llama.cpp: 98-100% agreement at Q8_0 (Q6_K was 94-97%, hence Q8_0). If you have an older file, replace it.
- **An Albanian typing model v2** (more realistic c/q/ç, digraph and case-ending errors) may replace `sq_typing_v1.1`.

## Licences and data caveats

| File | Licence | Why | Caveat |
|---|---|---|---|
| `Greek-*-v4.bin` | MIT | Whisper is MIT; training data is CC0, CC-BY, public domain or our own synthetic clips | Keep the CC-BY credits (FLEURS, YODAS) in `ATTRIBUTION.md` |
| `Albanian-244-v2.bin` | MIT | Whisper is MIT; Common Voice and Spontaneous Speech are CC0 | 1,966 of its training clips come from `akadriu/albanian-asr-10k-processed`, and it was distilled from `Flutra/whisper-large-v3-turbo-sq-v2`. **Neither states a licence**, and the dataset does not say where its audio comes from |
| `el_typing_v1.1`, `sq_typing_v1.1` | MIT | Trained from scratch; weights are our own work | Trained on web, Wikipedia, OpenSubtitles and (Greek) Reddit text; see below |
| `Greek-main_el.dict` / `.combined` | GPL-3.0 | Includes forms from Helium314's `main_el2`, which is GPL-3.0 (from dim-geo/greekdictionary) | `.combined` is the corresponding source |
| `Albanian-main_sq.dict` / `.combined` | MIT | Word list and frequencies counted from our corpora and our own lists | — |
| `Cleanup-v2-q4_0.gguf` | Apache-2.0 + Gemma Terms of Use | Qwen3-1.7B is Apache-2.0; part of the Greek targets were generated by Gemma 3, which can make it a Gemma "Model Derivative" | Gemma use restrictions pass on to you; see `NOTICE`. Source messages include OpenSubtitles and Reddit text |

What the licences on the training text mean for model weights is not settled law. In plain terms:

- **Wikipedia (CC BY-SA).** The common position, which most published language models take, is that trained weights
  are not an adaptation of the text, so share-alike does not reach them. Wikipedia is credited in `ATTRIBUTION.md`.
- **FineWeb-2 (ODC-By 1.0)** is also subject to Common Crawl's Terms of Use; the underlying web pages keep their
  own copyrights.
- **OpenSubtitles (via OPUS)** is text from opensubtitles.org. OPUS says it does not own the text and removes
  material on request, and its terms ask users to link to opensubtitles.org. The subtitles are fan transcriptions
  and translations of copyrighted films and shows, and nobody licensed them for model training. Many public models are
  trained on this corpus, for example Helsinki-NLP's OPUS-MT translation models. The risk is low for small models
  that cannot reproduce the text, but not zero.
- **Reddit (via IMISLab's GreekReddit: about 1% of the Greek typing model's training mix, and about 15% of the Greek source messages for the cleanup model).** The dataset is published
  under Apache-2.0, but Reddit's terms forbid using Reddit content to train models without Reddit's permission, and
  the posts belong to their authors.

Takedown or licence questions: open an issue in this repository.
