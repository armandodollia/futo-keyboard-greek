# Dictation cleanup LM

Trains the small on-device model behind **Settings → Voice input → Dictation cleanup**. It turns raw voice
transcripts into sendable text in two modes:

- **light:** punctuation, capitals, removes fillers, applies self-corrections ("at five, no sorry, at six" → "at six"),
  and otherwise keeps the speaker's words.
- **rambler:** rewrites the transcript as a short, clear message.

Both modes keep the language, slang and swearing exactly as spoken.

The shipped model is Qwen3-1.7B with a LoRA fine-tune for Greek, Albanian and English, exported as a Q4_0 GGUF
(about 1 GB). This folder is the generic version of that pipeline. Adding a language means writing one config file,
`languages/<code>.json`.

## How it works

1. **Source messages** (`sample_sources.py`): short informal texts in the language, 6 to 60 words each, taken from
   subtitles, chat or comments. These are the *clean* side of every pair.
2. **Pairs** (`gen_pairs.py`): each message is turned into messy dictation **by rule**: lowercase, no punctuation,
   fillers, repeated words, false starts and a self-correction. That gives a *light* pair whose target is the original
   message, so it is perfect by construction. A **teacher LLM** writes the *rambler* target, and a rewrite is kept only
   if it keeps the swearing, has a sane length and contains no chatter. 15% of inputs are left clean, so the model
   learns to leave good text alone.
3. **Trailing corrections** (`augment_corrections.py`): extra pairs in which the speaker corrects a word at or near the
   end ("…the boy is sick, sorry I meant the girl"). Numbers get their own examples. The first model got these wrong.
4. **Training** (`train_cleanup.py`): a LoRA (rank 32, all projections, loss on the answer only) on one short system
   line. It is the app's exact prompt (see below).
5. **Evaluation** (`eval_cleanup.py`): per language on held-out pairs: similarity to the target, swear retention,
   whether the output stayed in the language, and held-out trailing corrections.
6. **Export** (`export_gguf.py`): converts to GGUF and quantizes to Q4_0. `q4_check.py` checks the quantized file
   through llama-server with the app's raw prompt.

`pipeline.py` runs every step.

## The prompt (must match the app)

The app sends exactly this, and the model is trained on nothing else:

```
system: {task}. Language: {LanguageName}. Never translate. Keep slang and swearing exactly. Output only the text.
task:   "Rewrite as a short, clear message"                                                           (rambler)
        "Clean up: punctuation, capitals, remove fillers, apply self-corrections, keep the words"     (light)
```

It uses the Qwen3 chat template with thinking disabled, which puts an empty `<think>\n\n</think>\n\n` at the start of
the assistant turn. `common.py` builds the prompt (`system_prompt`, `raw_prompt`), and `train_cleanup.py --dry-run`
asserts that the tokenizer's chat template produces the same string. **If you change the wording in either place,
change it in both and retrain.**

### Language names

The app derives `LanguageName` from the voice-input language code:

```kotlin
java.util.Locale(code).getDisplayLanguage(java.util.Locale.ENGLISH)   // el -> Greek, sq -> Albanian, en -> English
```

`common.lang_name(code)` returns the same string. It uses a table of the CLDR English names that this call returns,
checked against JDK 17 for all 121 codes in the table, including the legacy Java codes `iw`, `in` and `ji`. Codes not
in the table fall back to `babel` (CLDR data, if installed), and then to the code itself, as Java does. Rules:

- Use the **bare ISO 639 code** (`pt`, not `pt_BR`). Java treats `pt_BR` as an unknown language and returns `pt_br`,
  so the model would see "Language: pt_br".
- The name comes from the device's CLDR data. Most names never change, but a few have changed between CLDR versions
  (`bn` is "Bangla" in recent data and "Bengali" in older data). If your language is one of these, check
  what your phone returns and train on that name.
- Quick check: `python -c "import common; print(common.lang_name('pt'))"`.

## Adding a language, step by step

You need a CUDA GPU for training (24 GB is comfortable; see the timings below), any machine for the other steps, and an
OpenAI-compatible LLM server for the teacher.

```sh
pip install -r requirements.txt
```

1. **Write the config.** Copy `languages/template.jsonc` to `languages/<code>.json` and fill in, **with a native
   speaker**:
   - fillers
   - correction phrases (inline and trailing)
   - number words two to ten
   - swear-word stems
   - a `detect` regex
   - a few hard test cases
   
   `el.json`, `sq.json` and `en.json` are complete examples.
2. **Get source messages** (about 8,000):
   ```sh
   # from OpenSubtitles (streams the OPUS monolingual file; oversamples lines with swearing)
   python sample_sources.py --lang pt --out work/sources_pt.txt --n 8000 --opus
   # or from your own corpora (FILE:SHARE[:join]; :join merges 1-3 dialogue lines) plus hand-written slang
   python sample_sources.py --lang pt --out work/sources_pt.txt --n 8000 \
       --source subtitles.txt:0.45:join --source forum.txt:0.3 --source web.txt:0.25 --always slang.txt
   ```
   Read a hundred lines. Swearing and slang must be there, or the model will learn to drop them.
3. **Start a teacher** and generate pairs. See "Teacher options" below.
   ```sh
   python gen_pairs.py --lang pt --sources work/sources_pt.txt --out work/pairs_pt.jsonl \
       --url http://localhost:8080/v1 --workers 6
   ```
   Read some rambler rewrites. If the teacher censors, translates or chatters, pick another teacher.
4. **Add correction pairs:** `python augment_corrections.py --work work`
5. **Train.** Include the existing languages' pairs in `work/` to get one shared model, which is what the app's
   *Download on-device model* installs. Alternatively train a per-language adapter (see "Per-language adapters" below).
   ```sh
   python train_cleanup.py --work work --out work/out/cleanup --dry-run   # checks prompts, no GPU needed
   python train_cleanup.py --work work --out work/out/cleanup
   ```
6. **Evaluate:** `python eval_cleanup.py --work work --hard work/out/cleanup`. Compare with the reference results below.
7. **Export:** `python export_gguf.py --llama-cpp ../llama.cpp --model work/out/cleanup --out work/cleanup-q4_0.gguf`,
   then check it: `llama-server -m work/cleanup-q4_0.gguf --port 8099` and `python q4_check.py --langs pt`.

Or run it all: `python pipeline.py --langs el,sq,en,pt --teacher-url http://localhost:8080/v1 --llama-cpp ../llama.cpp`.
Steps are `sources,pairs,augment,train,eval,export`. Select some with `--steps`.

`work/` holds all data, checkpoints and outputs. It is gitignored. **Do not commit datasets, generated pairs or
weights.**

### Teacher options

`gen_pairs.py --teacher openai` talks to any server with `/v1/chat/completions`:

- **llama-server** (llama.cpp): `llama-server -m <model>.gguf --port 8080 -np 6 -c 16384`. Use `--workers 6` to match
  the slot count. For Qwen3-style thinking models, add `--no-think`.
- **vLLM, Ollama** (`http://localhost:11434/v1`), **LM Studio**, or a hosted API: pass `--url`, `--model`, and
  `--api-key` or `$OPENAI_API_KEY`.
- **MLX on a Mac:** `--teacher mlx --model <mlx model>` (needs `pip install mlx-lm`).

Use the strongest model you can run that is fluent in the language and does not refuse or soften swearing. Test it by
hand on a few messages with insults first. With a local llama-server and 6 parallel slots, we generated about
100 to 175 messages a minute: 4,320 Greek messages took 41 min, and 15,720 Albanian+English messages took 90 min.
Rejected rewrites were well under 2%.

## Hardware and time (reference)

Measured on one RTX 4090 with bf16, batch 16 × 2 accumulation, max length 384, gradient checkpointing:

| run | pairs | epochs | steps | time |
| --- | --- | --- | --- | --- |
| v1: el+sq+en, from Qwen3-1.7B | 46,114 | 2 | 2,883 | ~94 min |
| v2: continued from v1 with correction pairs + ~30% of the old pairs (`--base v1 --epochs 1 --old-frac 0.3`) | 31,644 | 1 | 989 | ~32 min |

Evaluation with bf16 transformers takes about 1.5 s per item on the same GPU. Training on CPU is not practical. The
data steps need no GPU.

## Results so far (reference)

Shared el+sq+en model v2. The scores are on held-out pairs that were never trained on:

| metric | v2 |
| --- | --- |
| similarity to target (1 − CER) | 84.5% |
| swear-word retention | 93.1% |
| language kept | 99.3% |
| held-out trailing self-corrections (1 − CER vs original) | 87.8% |

The hard cases show what is still weak:
- Agreement after a corrected word: "ο μικρός είναι άρρωστος όχι συγγνώμη η μικρή" → "η μικρή είναι άρρωστος".
- Some number corrections in Albanian: "në orën pesë jo më fal në gjashtë" → "në gjashtë pesë".

These numbers came from the original scripts. The generic `eval_cleanup.py` scores every language separately, and its
correction test generator was fixed (see the `augment_corrections.py` docstring), so expect small differences.

## Using the model

- **On the phone:**
  1. Copy the `.gguf` to the phone.
  2. Open Settings → Voice input → Dictation cleanup → **Import on-device model (.gguf)**, then pick the local or
     parallel strategy.
  
  Import takes a **merged** model. Per-language adapters are installed through the download index instead (see
  below).
- **Remotely:** run `llama-server -m cleanup-q4_0.gguf --host 0.0.0.0 --port 8080` (or the f16 file, on a PC GPU) and
  enter its URL as the remote server under Dictation cleanup. Keep it on a trusted network or behind a VPN, because the
  API key is optional.
- **Per-language adapters:** see the next section.

## Per-language adapters (base + one small file per language)

The app can also run one shared base with a LoRA adapter per language (patch 0005): it loads the adapter for the
dictation language. The published adapters (el, sq, en: rank 16, alpha 32, all linear layers, 2 epochs, same data and
prompt as the shared model) are trained on the **stock `Qwen/Qwen3-1.7B`**, not on the shared cleanup model, and the
published base is stock Qwen3-1.7B in Q4_0. They match the shared model within noise (see `models/README.md`).
Adding a language this way needs no retraining of the others and ships a 33 MB file instead of a new 1 GB model.

1. Steps 1-4 above for the new language (config, sources, pairs, corrections).
2. **Train on this language only, from the stock base:**
   ```sh
   python train_cleanup.py --work work --lang pt --rank 16 --out work/out/cleanup-pt
   ```
   Keep the default `--base Qwen/Qwen3-1.7B`: an adapter only works on the exact base it was trained on. This writes
   the merged model to `work/out/cleanup-pt` and the adapter to `work/out/cleanup-pt-adapter`.
3. **Evaluate** against the shared model: `python eval_cleanup.py shared=work/out/cleanup pt=work/out/cleanup-pt-adapter`.
4. **Convert the adapter** to an f16 GGUF (llama.cpp's `convert_lora_to_gguf.py`, same llama.cpp version as the app,
   see `patches/llama.cpp.version`):
   ```sh
   python export_gguf.py --llama-cpp ../llama.cpp --adapter work/out/cleanup-pt-adapter --base Qwen/Qwen3-1.7B \
       --out cleanup-pt-lora-f16.gguf
   # or directly:
   python ../llama.cpp/convert_lora_to_gguf.py work/out/cleanup-pt-adapter --base <Qwen3-1.7B folder> \
       --outtype f16 --outfile cleanup-pt-lora-f16.gguf
   ```
   If your `convert_lora_to_gguf.py` lacks `--base-model-id`, download the base model and pass its folder as `--base`.
5. **Check it in llama.cpp** with the published base:
   `llama-server -m qwen3-1.7b-q4_0.gguf --lora cleanup-pt-lora-f16.gguf --port 8099`, then
   `python q4_check.py --langs pt`.
6. **Publish:** add a `cleanup-adapter` entry with `"locale": "pt"` to `models/index.json` (see the index section in
   `models/README.md`). The app shows a "Per-language models: base + Portuguese" row for it.

The base file itself is stock Qwen3-1.7B: `convert_hf_to_gguf.py` → f16 → `llama-quantize … Q4_0`. The Hugging Face
checkpoint stores the tied `lm_head.weight` explicitly; drop it before converting (it is identical to
`embed_tokens`), or the GGUF carries a second 622 MB (f16) output matrix.

## Files

| file | purpose |
| --- | --- |
| `common.py` | language names, the app's prompt, config loading, messy-dictation rules, swear counting, model loading |
| `languages/*.json` | per-language fillers, corrections, numbers, swear stems, detect regex, hard cases. `template.jsonc` is the template |
| `sample_sources.py` | source messages from local corpora and/or OPUS OpenSubtitles |
| `gen_pairs.py` | light + rambler pairs with an OpenAI-compatible or MLX teacher |
| `augment_corrections.py` | trailing self-correction pairs |
| `train_cleanup.py` | LoRA training, merged model + adapter |
| `eval_cleanup.py` | per-language evaluation, shared vs per-language models |
| `export_gguf.py` | GGUF f16 → Q4_0, or LoRA GGUF |
| `q4_check.py` | quality/speed check of a GGUF through llama-server |
| `pipeline.py` | runs every step |
