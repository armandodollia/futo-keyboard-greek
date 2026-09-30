# FUTO Keyboard Polyglot

An automatically built, **modified** version of [FUTO Keyboard](https://keyboard.futo.org)
([source](https://github.com/futo-org/android-keyboard)) that makes FUTO's offline smart features work for languages
beyond English, plus the tooling to train models for your own language. It is not an official FUTO release.
FUTO Keyboard is licensed under the FUTO Source First License 1.1 (`LICENSE-FUTO.md`), which allows modification and
free, non-commercial sharing.

- **Transformer autocorrect and prediction in any alphabet:** Greek, Albanian (ë, ç), Cyrillic and others (patch 0001).
- **Dictation cleanup:** after voice input, an LLM tidies the transcript or turns rambling into a short message, on a
  server you run, on the phone itself, or both in parallel (patches 0002 and 0003).
- **Training tooling** (`training/`): the typing model, voice model, dictionary and cleanup model for a new language.

Languages built with it so far: Greek and Albanian (typing model, voice model, dictionary) and a Greek/Albanian/English
cleanup model.

## What's changed

### 0001: Unicode keystrokes

`patches/0001-unicode-keystrokes.patch` changes one file, `native/jni/org_futo_inputmethod_latin_xlm_LanguageModel.cpp`.

Stock FUTO passes only `a–z` keystrokes to the transformer language model, so a model for another alphabet can't
complete or correct what you type. With the patch:

- **Any letter works.** Typed letters are read as Unicode characters, not bytes.
- **Keys map to the model's letters.** For a model without FUTO's `<CHAR_A>`…`<CHAR_Z>` tokens, each letter key maps to
  the model's own single-letter token.
- **Capitalisation is Unicode-aware.**
- **No crash on unknown letters.** A letter the model has no token for no longer crashes the keyboard.
- **FUTO's own (English) models behave exactly as before.**

It is based on [stamchry/android-keyboard@5795368](https://github.com/stamchry/android-keyboard/commit/5795368), cut down
to what the feature needs. To use a non-English typing model, also turn on Settings → Developer → "Allow transformer
models on non QWERTY layouts".

### 0002: Dictation cleanup

`patches/0002-dictation-cleanup.patch`. After voice input, the transcript can be cleaned (light: punctuation, fillers,
self-corrections) or rewritten (rambler: a short, clear message) by an LLM. It is set up under Settings → Voice input
→ Dictation cleanup:

- **Remote server:** any OpenAI-compatible server, e.g. llama-server on your own PC over Tailscale. The API key is
  optional, so it can be left blank on a trusted tailnet.
- **Parallel with the on-device model**, with a configurable strategy: prefer remote, fastest wins, remote only, or
  local only.
- **Swearing and slang are never softened.** A result that censors the speaker is rejected, and the next option or the
  raw transcript is used instead.
- **Keep original:** tap during "Cleaning up…" to keep the raw transcript.
- **Any language:** the prompt names the voice input language in English (from Android's locale), which is also the
  name the cleanup models are trained with.

The feature is off by default. The build declares the INTERNET permission, which it uses only to contact the server
you configure.

### 0003: On-device cleanup

`patches/0003-on-device-cleanup.patch`:

- **Separate native library:** `libllmcleanup.so` is built from a pinned modern llama.cpp (`patches/llama.cpp.version`,
  fetched at build time) for 64-bit devices, with hidden symbols so it never clashes with FUTO's own copy.
- **Import a model:** under Dictation cleanup, *Import on-device model (.gguf)* loads a fine-tuned cleanup model, e.g. a
  Qwen3-1.7B at Q4_0 (~1 GB).
- **Fully offline** when used alone, or as the backup for the remote server.

### The build

- **Installs as a separate app:** `org.futo.inputmethod.latin.polyglot`, named "FUTO Keyboard Polyglot", next to the
  official app. Pick it in Android's keyboard settings.
- **Turns off FUTO's own update check,** because updates come from this repo.

## Training models for your language

See [`training/README.md`](training/README.md). Each component has its own guide and a per-language config, with the
Greek and Albanian configs as worked examples:

| Component | Folder | What you get |
|---|---|---|
| Typing model | `training/typing-lm/` | Next-word prediction and autocorrect (`.gguf`) |
| Voice model | `training/voice/` | Whisper fine-tune for voice input (`.bin`) |
| Dictionary | `training/dictionary/` | Word list with slang; swear words follow "Block offensive words" (`.dict`) |
| Cleanup model | `training/cleanup-lm/` | Dictation cleanup LLM (`.gguf`) |

## How builds happen

`.github/workflows/build.yml` runs daily and builds every new upstream **stable** release once, skipping `-rc` builds:

1. Fetches FUTO's source at the release tag.
2. Applies `patches/*.patch`.
3. Builds and signs the APK with the repo's key (the `KEYSTORE_B64`, `KEYSTORE_PASSWORD` and `KEY_ALIAS` secrets).
4. Publishes a release named `<upstream>-polyglot.<rev>`.

It also rebuilds when the patches change, which bumps `<rev>`, or by hand under Actions → "Build patched FUTO
Keyboard" → Run workflow. `tools/mac-build.sh <tag>` does the same build locally.

**If FUTO changes the code so a patch no longer applies, the run fails and GitHub emails the repo owner.** Nothing is
published, so phones stay on the last working build.

## Installing and updating with Obtainium

Tap on the phone: https://apps.obtainium.imranr.dev/redirect?r=obtainium://add/https://github.com/armandodollia/futo-keyboard-polyglot

Or, in Obtainium: Add App → `https://github.com/armandodollia/futo-keyboard-polyglot` → Add. No token is needed.
Keep "Include prereleases" off. Each release has one APK plus a `.sha256` checksum.

The signing key never changes. Updates only install over a build signed with the same key, which is why builds
from anywhere else, including FUTO's own APK, install as a separate app rather than as an update.

Releases up to `0.1.30-greek.4` were built as "FUTO Keyboard (Greek patch)" (`org.futo.inputmethod.latin.greek`).
That app doesn't update to Polyglot: install Polyglot, re-import your models and dictionaries, then uninstall the old
app.
