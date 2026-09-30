# Training models for your language

FUTO Keyboard Polyglot uses four kinds of model. They are independent: build only the ones your language needs, in
any order. Each folder has a step-by-step guide, a `pipeline.py` (or `build_dict.py`) that runs everything, and a
per-language config in `languages/`. The Greek (`el`) and Albanian (`sq`) configs are the worked examples, and
`template.*` is the commented starting point.

| Folder | Builds | Install in the app | Hardware | Time (reference) |
|---|---|---|---|---|
| [`dictionary/`](dictionary/README.md) | `.dict` word list with frequencies; swear words flagged so "Block offensive words" controls them | Languages & Models → the language → dictionary | Any PC, Java 17 | Minutes |
| [`typing-lm/`](typing-lm/README.md) | `.gguf` transformer for next-word prediction and autocorrect | Languages & Models → the language → import model; Developer → "Allow transformer models on non QWERTY layouts" | NVIDIA GPU (CUDA) | ~2 h 15 min for 1.5B tokens on an RTX 4090 |
| [`voice/`](voice/README.md) | `.bin` Whisper model for voice input (tiny/base/small) | Languages & Models → Voice input → import | NVIDIA GPU, 12 GB+ for small | ~2.5 h for a Greek small on a 4090 |
| [`cleanup-lm/`](cleanup-lm/README.md) | `.gguf` dictation-cleanup LLM (Qwen3-1.7B LoRA) | Voice input → Dictation cleanup → Import on-device model | NVIDIA GPU, 24 GB; a teacher LLM on any OpenAI-compatible server | ~1.5 h training, plus pair generation |

Apple Silicon (MPS) works for the smaller scripts but was about 40× slower than CUDA for the typing model, so a CUDA
GPU is recommended for training.

## Suggested order for a new language

1. **Dictionary.** Quickest win: it improves suggestions and spell-checking in stock FUTO too.
2. **Voice.** Worth it if FUTO's multilingual voice models do poorly in your language. Common Voice is the default
   data source.
3. **Typing model.** The biggest change to typing. It needs the patched keyboard: stock FUTO passes only a–z
   keystrokes to the model.
4. **Cleanup model.** Add your language to the existing multilingual model, or train a per-language adapter.

## Things that apply to every component

- **Your layout decides the typing model.** The typing model is trained on the geometry of one FUTO keyboard layout.
  `typing-lm/futo_layout.py` reads it from [futo-keyboard-layouts](https://github.com/futo-org/futo-keyboard-layouts).
- **Check what you train on.** The READMEs say which public corpora each pipeline downloads and what their licences
  mean if you publish a model. Subtitle and Reddit text are the risky ones. [`../models/`](../models/README.md) shows
  how the Greek and Albanian models were credited and licensed.
- **Get a native speaker to review the slang and offensive-word lists.** They decide what gets suggested and what
  "Block offensive words" hides.
- **Language names:** the cleanup model is prompted with the language's English name as Android reports it
  (`Locale.forLanguageTag(code).getDisplayLanguage(Locale.ENGLISH)`). `cleanup-lm` uses the same rule, so the prompts
  match.

Contributions of configs and models for new languages are welcome: open an issue or a pull request.
