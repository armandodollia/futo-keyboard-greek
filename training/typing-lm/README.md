# Typing models for FUTO Keyboard Polyglot

This toolkit trains a small transformer for one language that does **next-word prediction and autocorrect/completion**
in FUTO Keyboard. It is the same kind of model as FUTO's English one. Patch 0001 (`patches/0001-unicode-keystrokes.patch`)
lets the keyboard feed such a model keystrokes in any alphabet. Without it, only `a–z` reach the model.

Two languages are ready: `el` (Greek) and `sq` (Albanian). Adding a language means writing one config file,
`languages/<code>.yaml`. Everything else is generic.

## How the model works (short)

- **Tokenizer:** SentencePiece unigram, 16k pieces. Pieces end in `▁` (FUTO's `inverted_space`). Every letter of the
  alphabet is its own piece, because the keyboard maps each key to that letter's token.
- **Model:** Llama, 8 layers × 512, 35.5M parameters. GGUF sizes: 84 MB F16, 35 MB Q6_K.
- **Training mixes two kinds of example:**
  - Plain text, for next-word prediction.
  - Autocorrect examples, built exactly the way FUTO builds its input: `<s> context <XBU> key key … <XBC>` →
    `word▁`. Each keystroke is one input position, holding a blend of the embeddings of the 4 letter keys nearest the
    tap (`char_embed_mixing_v1`).
- **Taps are simulated on the language's real FUTO layout.** The simulator adds fuzzy taps, dropped, extra, swapped
  and doubled keys, the language's own misspellings, and half-typed words.
- **Diacritics:** a long-press diacritic is typed as the base letter, so the model learns to restore accents.

## Requirements

| | |
|---|---|
| GPU | **NVIDIA with CUDA recommended.** An RTX 4090 trains the default 1.5B tokens in ~2h15m (~185k tok/s). Apple MPS works, but was ~40× slower with this trainer (days). CPU is only for smoke tests. Run `--train-args "--probe"` first to see speed, peak memory and ETA. Out of memory: lower `--batch` (e.g. 128). |
| Disk | ~10–25 GB per language: raw text (FineWeb-2 is capped at `max_mb`, ~3 GB by default), cleaned text and token files. Big subtitle dumps add more (Greek: 11 GB raw). |
| RAM | 16 GB+. Tokenizer training on 1.5M sample lines is the peak (4M lines needed > 24 GB). |
| Python | 3.10+: `pip install -r requirements.txt`. Install the CUDA build of torch first. |
| Quantizing | `llama-quantize` from [llama.cpp](https://github.com/ggml-org/llama.cpp) on PATH, **or** `pip install llama-cpp-python`. The pip route also lets the pipeline check the files in llama.cpp. |

## Quick start (an existing language)

```sh
cd training/typing-lm
pip install -r requirements.txt
python pipeline.py --lang sq            # download, prepare, train, evaluate, export, quantize -> work/sq/
```

The result is `work/sq/sq_typing_Q6_K.gguf`. Stages can be run alone, e.g. `--stages train,evaluate`, and each one
reads the previous stage's output from `--work`. Everything is also logged to `work/<code>/pipeline.log`.

## Adding a language, step by step

1. **Copy the template.** Copy `languages/template.yaml` to `languages/<code>.yaml`. Use the ISO 639-1 code that FUTO
   uses for the language: it is written into the model, and the keyboard only offers the model for that language.
2. **Alphabet.** Set `letters`, a regex class that includes capitals and accented letters. Also set `alphabet`,
   `sentence_start` and `sentence_end`.
3. **Layout.** Find your language's file in
   [futo-keyboard-layouts](https://github.com/futo-org/futo-keyboard-layouts) and run
   ```sh
   python futo_layout.py LatinScript/albanian.yaml     # path in that repo, a raw URL, or a local file
   ```
   It prints the `layout:` block:
   - **Rows:** the base letter of every key, left to right, plus the row's x offset in key widths. FUTO centres rows,
     so offset = (keys in the widest row − keys in this row) / 2.
   - **`own_keys`:** accented letters that have their own key.
   - **Long-press list:** every long-press alternate, printed as comments. These are typed as the base letter.

   Check anything unusual by hand, such as custom key widths or a split layout. If your language has several
   layouts, use the one most people type on.
4. **Typing errors.** List your language's common misspellings in `typing_errors`: phonetic confusions, digraphs,
   letters people type without their accent, wrong endings. `sq.yaml` shows every rule type. The generic tap noise is
   always added on top. Keep the probabilities modest, since most words should be typed right.
5. **Data.** Set up `sources`:
   - **FineWeb-2:** set `dataset` to the FineWeb-2 config for your language (`<iso639-3>_<Script>`, e.g. `por_Latn`;
     see the `data/` folder of [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)).
   - **Wikipedia and OpenSubtitles:** use the language code by default. Override with `wiki:` or `opus:` if needed.
   - **Informal text:** add anything that exists for your language (Reddit dumps, comment datasets, offensive-language
     sets) with `from: huggingface`.
   - **Your own text:** put slang, chat phrases or insult lists in `languages/extra/<code>/*.txt`, one sentence per line
     (`from: local`). Held-out test sentences go in `languages/extra/<code>/test/*.txt` (`eval.text`), which the
     training glob doesn't read. Commit these files only if you wrote them or may share them.
6. **Weights.**
   - `sp` is each source's share of the tokenizer sample. Weight informal text up, so that slang gets its own pieces.
   - `mix` is each source's share of training. A source is seen `mix × total tokens / source tokens` times: keep small
     sources at ~15–75 repeats, not hundreds. `prepare` writes each source's token count to `work/<code>/tok/stats.json`.
7. **Run the first stages.**
   ```sh
   python pipeline.py --lang xx --stages download,prepare
   ```
   Check `work/xx/tok/stats.json` (see the checklist below). Adjust `mix` from the token counts.
8. **Train.**
   ```sh
   python pipeline.py --lang xx --stages train --train-args "--probe"     # speed + ETA, ~1 minute
   python pipeline.py --lang xx --stages train,evaluate,export,quantize
   ```
9. **Optional fine-tune.** Albanian v2 did this: v1 was fine-tuned with extra error rules, which are tagged `v2` in
   `sq.yaml`.
   ```sh
   python pipeline.py --lang sq --work work/sq-v2 --tok work/sq/tok --init work/sq/model --error-tags v2 \
       --tokens 3e8 --train-args "--lr 3e-4" --stages train,evaluate,export,quantize
   ```

**Smoke test on a CPU,** to check the plumbing in minutes with a tiny model. Put a few thousand lines per source in
`work/xx-smoke/raw/<source>.txt`, or run `download` first. The default vocabulary of 16k is too big for 5,000 lines, so
point `--lang` at a copy of the config with `tokenizer.vocab: 1500`:
```sh
python pipeline.py --lang my-xx-copy.yaml --work work/xx-smoke --stages prepare,train,evaluate,export,quantize \
    --prepare-args "--max-lines 5000" --tokens 30000 \
    --train-args "--layers 2 --hidden 128 --ffn 256 --batch 8 --seq 64 --workers 0 --device cpu"
```

## Install it in the app

1. Install FUTO Keyboard Polyglot. Add your language and its layout under **Settings → Languages & Models**.
2. **Settings → Languages & Models** → your language → **import the model** → pick `<code>_typing_Q6_K.gguf`.
   The keyboard matches the file's `keyboardlm.languages` against the language.
3. **Settings → Developer → "Allow transformer models on non QWERTY layouts"**: turn it on. Without it, FUTO ignores the
   model on any layout that isn't QWERTY, which includes Greek, QWERTZ and most others.
4. Type something. Suggestions should appear before you type, and typos should be corrected as you type.

## Sanity checklist

**After `prepare`** (`work/<code>/tok/stats.json`):
- [ ] `single_letter_pieces_missing` is `[]`. `prepare` stops with an error if a keyboard letter has no piece.
- [ ] `examples` split informal words sensibly: `vlla▁`, not `v l l a▁`.
- [ ] Every source has a plausible number of lines and tokens. A near-empty source usually means the `letters` filter
      or `min_script_share` is wrong (or the download failed: see the log).
- [ ] `*.clean.txt` spot check: real sentences in your language. No markup, and no other language dominating.

**During `train`:**
- [ ] `--probe` gives the throughput you expect. A 4090 does ~185k tok/s.
- [ ] The loss falls steadily. Greek ended at ~3.44 (LM + autocorrect combined).

**After `evaluate`.** The reference numbers are top-1 / top-3, on held-out text:

| | next-word | clean | typos | partial |
|---|---|---|---|---|
| Greek v1 | 16 / 27% | 88 / 94% | 67 / 81% | 45 / 67% |
| Albanian v1 | 28 / 41% | 91 / 96% | 75 / 87% | 55 / 73% |

- [ ] `clean` below ~80% top-1 usually means a layout or diacritic problem. Check `layout.rows` and `own_keys`.

**After `export` and `quantize`:**
- [ ] `export` prints `check: ... tokenizer bytes ok=True ... argmax agree=100.0%`.
- [ ] With llama-cpp-python installed, `check_gguf.py` gives ~100% argmax agreement for F16 and ≥ ~95% for Q6_K. Much
      lower means the export is broken.

**On the phone:**
- [ ] The model shows up for the language and the import doesn't fail.
- [ ] Next-word suggestions appear after a space.
- [ ] A word typed without accents gets them back, and a typo'd word is corrected.
- [ ] Capitalised and ALL-CAPS words are suggested in the same case.
- [ ] Typing letters the model has no token for (digits, other scripts) doesn't crash the keyboard.

## Files

| | |
|---|---|
| `pipeline.py` | entry point: download → prepare → train → evaluate → export → quantize |
| `languages/*.yaml` | per-language config: `el`, `sq`, `template` (documents every key) |
| `langconfig.py` | loads a config; layout geometry, keystroke / typing-error simulator, tap mixing |
| `download.py` | FineWeb-2, Wikipedia, OpenSubtitles (OPUS), any Hugging Face dataset, URLs |
| `prepare.py` | cleaning, dedup, SentencePiece, `.bin`/`.idx` token files |
| `train.py` | the trainer (LM + autocorrect examples) |
| `evaluate.py` | next-word / autocorrect top-1 and top-3, plus optional misspelling-pair benchmark |
| `export.py` | FUTO GGUF writer (metadata, tokenizer, RoPE permutation) with a round-trip check |
| `check_gguf.py` | runs the GGUF in llama.cpp and compares with PyTorch |
| `futo_layout.py` | prints the `layout:` block from a FUTO layout file |

## Licences and caveats

- **Data licences.** FineWeb-2 is ODC-By and Wikipedia is CC BY-SA. **OpenSubtitles text is copyrighted.** It is the
  best informal dialogue available and fine for a model you use yourself. Think twice before publishing a model
  trained on it, or drop that source for a model you share. Check the licence of every Hugging Face dataset you add.
- **Export fix.** `export.py` reorders the attention Q/K weights for llama.cpp's RoPE layout, as llama.cpp's own
  converter does. Some earlier community exporters skipped this. Their GGUFs load and suggest words, but with
  scrambled position information. In a test model with sharp attention, only 40% of llama.cpp's top predictions
  matched PyTorch without the reordering, against 100% with it. If you have such a model, re-export its checkpoint
  with `export.py`. No retraining is needed.
- **Language code.** It must match the keyboard's. Regional variants (`pt_BR`) depend on how FUTO names the language.
  Check the language list in the app.
