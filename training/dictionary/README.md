# Dictionary builder

Builds a FUTO Keyboard dictionary (`main_<code>.dict`) for any language from plain text corpora. The dictionary gives
FUTO its word list and word frequencies: suggestions, autocorrect targets and swipe candidates all come from it.

`build_dict.py` is generic; everything language-specific is in `languages/<code>.json`. Greek (`el.json`) and
Albanian (`sq.json`) are the configs our released dictionaries were built with. `languages/template.jsonc` is a
commented template for a new language.

## What it does

1. **Counts words** in each corpus with the config's `word_regex`. Every source has a **boost**: informal text is
   scarce, so subtitles, chat, offensive-comment datasets and slang lists are weighted up until chat vocabulary ranks
   where it would in real typing.
2. **Picks one casing per word.** A word is stored capitalised only if at least 90% of its uses are capitalised
   (names, places); otherwise it is stored lowercase, and FUTO capitalises it when needed.
3. **Maps counts to AOSP frequencies:** `f = 255 + log(p) / log(1.15)`, clipped to `[min_f, max_f]` (30…250 by
   default). A word that is kept never gets `f=0`, because AOSP treats 0 as "never suggest".
4. **Flags vulgar words** with `possibly_offensive=true` and keeps their **normal frequency**. FUTO's
   *Block offensive words* setting then decides whether they are suggested, and either way they stay valid words that
   are never autocorrected into something else. Capitalised forms are treated as names and are never flagged.
5. **Optionally merges an existing AOSP wordlist** (`extra_wordlist`) at low frequency. Greek uses Helium314's
   `main_el2`, so valid rare inflected forms are not "corrected".
6. **Writes `<code>_wordlist.combined`** and, with `--compile`, compiles it to `main_<code>.dict`.

## Offensive-word lists need a native speaker

The offensive lists are the part most likely to be wrong, and mistakes go both ways:

- **Stems catch innocent words.** Greek `γαμ…` also matches γάμος (wedding), `μαλακ…` matches μαλακά (softly), and
  Albanian `kopil…` matches kopilot (co-pilot). English `cock` would flag cockpit and cockroach.
- **Stems miss variants:** spellings without accents, dialect forms, and slang spellings.

The configs therefore have exception lists (`not_offensive_stems`, `not_offensive_exact`). Every build prints the
full list of flagged words. **A native speaker should read that list, and a sample of what it does not flag, before a
dictionary is published.** Flag only vulgar swearing and slurs, not mild insults or everyday slang.

## Build

Requirements: Python 3.9+ (standard library only), and Java 11+ to compile.

```sh
# corpora in ./corpus/sq, named as in languages/sq.json (paths are relative to --data)
python build_dict.py --lang sq --data corpus/sq --out build --compile
# -> build/sq_wordlist.combined, build/main_sq.dict
```

- `--lang` takes a code (`languages/<code>.json`) or a path to a config.
- If `java` is not on `PATH`, set `JAVA_HOME`.
- The build prints the top words, every flagged word and the frequencies of the config's `look` words. Check all
  three.

Corpus sources we used: web text (FineWeb-2), Wikipedia, OpenSubtitles, Reddit (el), offensive-language datasets,
speech transcripts (Albanian Gheg, Mozilla Spontaneous Speech), and a few hundred hand-written slang sentences, one per
line. Corpora are not part of this repo. Any UTF-8 text works, with one sentence or message per line. A `.jsonl`
source counts one field of each row (`"field"`, default `"text"`).

## dicttool_aosp.jar

The compiler is AOSP's `dicttool`, in the prebuilt `dicttool_aosp.jar` that
[Helium314/aosp-dictionaries](https://codeberg.org/Helium314/aosp-dictionaries) ships. It was built by
[remi0s/aosp-dictionary-tools](https://github.com/remi0s/aosp-dictionary-tools) (Apache-2.0) from AOSP LatinIME.
It is **not** included here. `--compile` downloads it on first use into this folder (it is gitignored) and checks its
SHA-256 against the version we tested:

```
https://codeberg.org/Helium314/aosp-dictionaries/raw/branch/main/dicttool_aosp.jar
sha256 a8c5bd21f631ed0a92235d42d2fe83af5d70216172bf7e22781a9a946858237e
```

To compile by hand:

```sh
java -jar dicttool_aosp.jar makedict -s build/sq_wordlist.combined -d build/main_sq.dict
```

The jar's other commands, such as `info`, need a native library that is not included, but `makedict` needs only Java.

## Importing into FUTO Keyboard

1. Copy `main_<code>.dict` to the phone.
2. Open it with the keyboard: tap the file in a file manager, or share it, and pick FUTO Keyboard (the patched build
   is listed under its own name). FUTO asks what to import it as. Choose the dictionary for your language.
   Alternatively, go to Settings → Languages & Models, open the language, and import the dictionary there.
3. Optional: FUTO's *Block offensive words* setting, in the text prediction settings, hides the flagged words from
   suggestions.

The keyboard language must match the dictionary's `locale`, which is `sq` in the example above.

## The .combined format

```
dictionary=main:sq,locale=sq,description=Shqip (Albanian incl. Gheg + slang),date=1790000000,version=1
 word=dhe,f=234
 word=qifsha,f=187,possibly_offensive=true
```

All header fields are required. `description` must not contain `,` or `=`. Our dictionaries use `version=1`, which FUTO
loads fine. The AOSP-dictionaries notes recommend `version` > 18 for some other AOSP-based keyboards.
