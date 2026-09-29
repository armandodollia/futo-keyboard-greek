# FUTO Keyboard, Greek patch (personal build)

This is an automatically built, **modified** version of [FUTO Keyboard](https://keyboard.futo.org)
([source](https://github.com/futo-org/android-keyboard)). It is not an official FUTO release. FUTO Keyboard is
licensed under the FUTO Source First License 1.1 (`LICENSE-FUTO.md`), which allows modification and free,
non-commercial sharing.

## What's changed

`patches/0001-unicode-keystrokes.patch` changes one file, `native/jni/org_futo_inputmethod_latin_xlm_LanguageModel.cpp`.

Stock FUTO passes only `a–z` keystrokes to the transformer language model, so the model can't complete or correct
words typed in Greek, Albanian (ë, ç) or other alphabets. With the patch:

- **Any letter works.** Typed letters are read as Unicode characters, not bytes.
- **Keys map to the model's letters.** For a model without FUTO's `<CHAR_A>`…`<CHAR_Z>` tokens, each letter key maps to
  the model's own single-letter token.
- **Capitalisation understands Greek.** Upper/lowercase detection is Unicode-aware.
- **No crash on unknown letters.** A letter the model has no token for no longer crashes the keyboard.
- **FUTO's own (English) models behave exactly as before.**

It is based on [stamchry/android-keyboard@5795368](https://github.com/stamchry/android-keyboard/commit/5795368), cut down
to what the feature needs.
`patches/0002-dictation-cleanup.patch` adds **dictation cleanup**. After voice input, the transcript can be cleaned (light:
punctuation, fillers, self-corrections) or rewritten (rambler: a short, clear message) by an LLM. It is set up under
Settings → Voice input → Dictation cleanup:

- **Remote server:** any OpenAI-compatible server, e.g. llama-server on your own PC over Tailscale. The API key is
  optional, so it can be left blank on a trusted tailnet.
- **Local model:** an on-device model, coming in a later patch. Remote and local run **in parallel**, and the strategy
  is configurable: prefer remote, fastest wins, remote only, or local only.
- **Swearing and slang are never softened.** A result that censors the speaker is rejected, and the next option or the
  raw transcript is used instead.
- **Keep original:** tap during "Cleaning up…" to keep the raw transcript.

The feature is off by default. The build declares the INTERNET permission, which it uses only to contact the server
you configure.

The build also:

- **Installs as a separate app:** `org.futo.inputmethod.latin.greek`, named "FUTO Keyboard (Greek patch)", next to the
  official app. Pick it in Android's keyboard settings.
- **Turns off FUTO's own update check,** because updates come from this repo.

## How builds happen

`.github/workflows/build.yml` runs daily and builds every new upstream **stable** release once, skipping `-rc` builds:

1. Fetches FUTO's source at the release tag.
2. Applies `patches/*.patch`.
3. Builds and signs the APK with the repo's key (the `KEYSTORE_B64`, `KEYSTORE_PASSWORD` and `KEY_ALIAS` secrets).
4. Publishes a release named `<upstream>-greek.<rev>`.

It also rebuilds when the patches change, which bumps `<rev>`, or by hand under Actions → "Build patched FUTO
Keyboard" → Run workflow.

**If FUTO changes the code so a patch no longer applies, the run fails and GitHub emails you.** Nothing is
published, so your phone stays on the last working build.

## Updating with Obtainium

Tap on the phone: https://apps.obtainium.imranr.dev/redirect?r=obtainium://add/https://github.com/armandodollia/futo-keyboard-greek

Or, in Obtainium: Add App → `https://github.com/armandodollia/futo-keyboard-greek` → Add. No token is needed.
Keep "Include prereleases" off. Each release has one APK plus a `.sha256` checksum.

The signing key never changes. Updates only install over a build signed with the same key, which is why builds
from anywhere else, including FUTO's own APK, install as a separate app rather than as an update.