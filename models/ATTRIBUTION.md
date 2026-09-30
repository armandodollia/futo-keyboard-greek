# Attribution

The models in release `models-v1` build on the work below. Thank you to everyone involved. Licences are as stated by
each project when checked (September 2026); where a project states none, this is noted.

## Base models, teachers and tools

| Component | Used for | Licence |
|---|---|---|
| [OpenAI Whisper](https://github.com/openai/whisper) `whisper-small`, `-base`, `-tiny` | Starting weights of all voice models | MIT, © 2022 OpenAI |
| [OpenAI Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) and large-v3-turbo | Greek teacher: re-labelled the audiobooks, checked the YouTube subtitles and synthetic clips | MIT, © 2022 OpenAI |
| [Flutra/whisper-large-v3-turbo-sq-v2](https://huggingface.co/Flutra/whisper-large-v3-turbo-sq-v2) (fine-tune of `openai/whisper-large-v3-turbo` on Common Voice 19 sq) | Albanian teacher: knowledge distillation (soft targets) for `Albanian-244-v2`, and relabelling of the akadriu clips | **No licence stated** on the model card; the base model is MIT |
| [FUTO whisper-acft](https://github.com/futo-org/whisper-acft) | Audio-context fine-tuning (ACFT) method for all voice models | MIT, © FUTO |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | Conversion to ggml and q8_0 quantization | MIT, © The ggml authors |
| [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) (Resemble AI) | Synthetic informal Greek speech | MIT |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (OpenBMB) | Synthetic informal Greek speech | Apache-2.0 |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (Qwen team, Alibaba Cloud) | Base model of `Cleanup-v2` | Apache-2.0 |
| [Gemma 3](https://ai.google.dev/gemma) `gemma-3-12b-it`, as [`mlx-community/gemma-3-12b-it-4bit`](https://huggingface.co/mlx-community/gemma-3-12b-it-4bit) (Google) | Teacher: rambler-mode rewrites for part of the Greek cleanup pairs, generated on a Mac with MLX | [Gemma Terms of Use](https://ai.google.dev/gemma/terms); see `NOTICE` |
| [Gemma 4](https://ai.google.dev/gemma) `gemma-4-12b-it`, as [`unsloth/gemma-4-12b-it-GGUF`](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) `UD-Q4_K_XL` (Google) | Teacher: rambler-mode rewrites for the Albanian, English and the rest of the Greek cleanup pairs, served by llama-server | [Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF conversion and quantization (Q8_0, Q4_0), serving the teacher | MIT, © The ggml authors |
| [MLX / mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple) | Running the Gemma 3 teacher | MIT |
| [stamchry/greek-keyboard-lm](https://github.com/stamchry/greek-keyboard-lm) | FUTO GGUF metadata layout followed by our exporter (`training/typing-lm/export.py`); its Greek LM is the comparison baseline | Apache-2.0 |
| [stamchry/android-keyboard](https://github.com/stamchry/android-keyboard) | Basis of the Unicode keystroke patch the typing models need | FUTO Source First License 1.1 (fork of FUTO Keyboard) |
| [AOSP dicttool](https://android.googlesource.com/platform/packages/inputmethods/LatinIME/) | Compiling `.combined` word lists into `.dict` | Apache-2.0 |
| [Hugging Face Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), [SentencePiece](https://github.com/google/sentencepiece) | Training and tokenizers | Apache-2.0 |

## Speech data

| Dataset | Used for | Licence |
|---|---|---|
| [Common Voice 27](https://commonvoice.mozilla.org) (Mozilla Foundation), Greek and Albanian, via the [Mozilla Data Collective](https://datacollective.mozillafoundation.org) | Training (test/dev speakers and sentences excluded) and evaluation; some Greek voices were TTS references for the synthetic clips | CC0-1.0 |
| Common Voice Spontaneous Speech 5.0, Gheg Albanian (`aln`) (Mozilla Foundation), via the Mozilla Data Collective | `Albanian-244-v2` training (train split, 615 clips) and the Gheg test | CC0-1.0; the download terms also forbid re-hosting the data and identifying speakers (no audio is redistributed here) |
| [akadriu/albanian-asr-10k-processed](https://huggingface.co/datasets/akadriu/albanian-asr-10k-processed) | `Albanian-244-v2` training: 1,966 clips kept where the dataset label agrees with the teacher; labels are the teacher's transcripts | **No licence and no source stated** on the dataset card |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google), `el_gr` | Greek training, ACFT and the FLEURS test | CC-BY-4.0 |
| [YODAS](https://huggingface.co/datasets/espnet/yodas) (ESPnet / WAVLab, CMU), subset `el000` | Greek training on Creative Commons YouTube speech with its human subtitles; held-out videos form the YouTube test | CC-BY-3.0 |
| [LibriVox](https://librivox.org) Modern Greek audiobooks (20 books) | Greek training, transcribed by Whisper large-v3 | Public domain |

The YODAS audio comes from YouTube videos their uploaders published under CC-BY; credit for the speech goes to those
creators. The Greek video and clip IDs are listed in the Greek model repositories' `training/data-lists`. No audio is
redistributed with these models.

## Text data

| Dataset | Used for | Licence |
|---|---|---|
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (Hugging Face), `ell_Grek`, `als_Latn` (standard Albanian) and `aln_Latn` (Gheg) | Typing models, dictionary frequencies, cleanup source messages | [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/); also subject to [Common Crawl's Terms of Use](https://commoncrawl.org/terms-of-use) |
| [Wikipedia](https://www.wikipedia.org) (Wikimedia Foundation and contributors), via [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.el` and `.sq` | Typing models, dictionary frequencies | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and GFDL |
| [OpenSubtitles](https://opus.nlpl.eu/legacy/OpenSubtitles.php) (OPUS, University of Helsinki), monolingual el and sq (v2024) and en (v2018), from subtitles on [opensubtitles.org](http://www.opensubtitles.org/) | Typing models, dictionary frequencies, cleanup source messages (Greek, Albanian, English) | No licence of its own for the text: OPUS states it does not own it and honours takedown requests. The v2024 release is labelled ODC-By on Hugging Face |
| [GreekReddit](https://huggingface.co/datasets/IMISLab/GreekReddit) (IMISLab) | Greek typing model, dictionary frequencies, Greek cleanup source messages | Apache-2.0 (dataset); the posts come from Reddit, whose terms restrict ML training |
| [OffensEval 2020 Greek / OGTD](https://huggingface.co/datasets/strombergnlp/offenseval_2020) (Pitenis, Zampieri, Ranasinghe), config `gr` | Greek typing model, dictionary, cleanup source messages (swearing and insults) | CC-BY-4.0 |
| [Shaj](https://huggingface.co/datasets/strombergnlp/shaj) (Nurce, Keci, Derczynski) | Albanian typing model, dictionary, cleanup source messages (swearing and insults) | CC-BY-4.0 |
| Common Voice Spontaneous Speech Gheg transcripts (see above) | Albanian dictionary (Gheg vocabulary) | CC0-1.0 |
| [Helium314/aosp-dictionaries](https://codeberg.org/Helium314/aosp-dictionaries) `main_el2`, from [dim-geo/greekdictionary](https://github.com/dim-geo/greekdictionary) | Rare Greek word forms added at low frequency to `Greek-main_el.dict` | GPL-3.0, which is why the Greek dictionary is GPL-3.0 |
| Our own informal-sentence lists (Greek and Albanian slang, insults, swearing, Gheg forms) | Typing models, dictionaries, cleanup source messages | Our own work (MIT) |

Cleanup training pairs: the input side is generated by rule from the source messages above (lowercase, no
punctuation, fillers, repeats, self-corrections); the light-mode target is the original message; the rambler-mode
target is a teacher's rewrite (Gemma 3 or Gemma 4, see above).

## Evaluation only

| Dataset | Used for | Licence |
|---|---|---|
| [greta44/albanian-error-augmentation](https://huggingface.co/datasets/greta44/albanian-error-augmentation) | Albanian misspelling test (not used for training; its error types informed the synthetic typos) | Apache-2.0 |
| Common Voice 27 test/dev, FLEURS test, held-out YODAS videos, Spontaneous Speech Gheg test | Voice tests | as above |

## References

- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*, 2022 (Whisper).
- Ardila et al., *Common Voice: A Massively-Multilingual Speech Corpus*, LREC 2020.
- Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*, SLT 2022.
- Li et al., *YODAS: YouTube-Oriented Dataset for Audio and Speech*, ASRU 2023.
- Penedo et al., *FineWeb2: One Pipeline to Scale Them All — Adapting Pre-Training Data Processing to Every Language*, 2025.
- P. Lison and J. Tiedemann, *OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles*,
  LREC 2016.
- Pitenis, Zampieri and Ranasinghe, *Offensive Language Identification in Greek*, LREC 2020.
- Nurce, Keci and Derczynski, *Detecting Abusive Albanian*, 2021.
- Mastrokostas, Giarelis and Karacapilidis, *Social Media Topic Classification on Greek Reddit*, Information 15(9), 2024.
- Qwen Team, *Qwen3 Technical Report*, 2025.
- Gemma Team, Google, *Gemma 3 Technical Report*, 2025.
