"""Shared pieces of the dictation-cleanup pipeline: language names, the app's exact prompt, per-language config,
messy-dictation rules, swear counting, language check, and model loading for evaluation.

The prompt MUST stay byte-identical to the keyboard's (patches/0003-on-device-cleanup.patch): the model is trained on
this one short system line and nothing else, so any drift costs quality on the phone.
"""
import json
import pathlib
import re
import unicodedata

HERE = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------- language names (must match the app)
# The app names the language with Java: Locale(code).getDisplayLanguage(Locale.ENGLISH). These are the CLDR English
# names that call returns (checked with JDK 17; Android uses ICU with the same CLDR data). Legacy Java codes
# (iw, in, ji) give the same names as he, id, yi. Unknown codes come back unchanged, like Java does.
JAVA_ENGLISH_NAMES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese", "ast": "Asturian", "az": "Azerbaijani",
    "be": "Belarusian", "bg": "Bulgarian", "bn": "Bangla", "bo": "Tibetan", "br": "Breton", "bs": "Bosnian",
    "ca": "Catalan", "ckb": "Central Kurdish", "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German",
    "el": "Greek", "en": "English", "eo": "Esperanto", "es": "Spanish", "et": "Estonian", "eu": "Basque",
    "fa": "Persian", "fi": "Finnish", "fil": "Filipino", "fo": "Faroese", "fr": "French", "fy": "Western Frisian",
    "ga": "Irish", "gd": "Scottish Gaelic", "gl": "Galician", "gsw": "Swiss German", "gu": "Gujarati", "ha": "Hausa",
    "haw": "Hawaiian", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian",
    "hy": "Armenian", "id": "Indonesian", "ig": "Igbo", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "jv": "Javanese", "ka": "Georgian", "kk": "Kazakh", "km": "Khmer", "kn": "Kannada", "ko": "Korean",
    "ku": "Kurdish", "ky": "Kyrgyz", "la": "Latin", "lb": "Luxembourgish", "lo": "Lao", "lt": "Lithuanian",
    "lv": "Latvian", "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian",
    "mr": "Marathi", "ms": "Malay", "mt": "Maltese", "my": "Burmese", "nb": "Norwegian Bokmål", "nds": "Low German",
    "ne": "Nepali", "nl": "Dutch", "nn": "Norwegian Nynorsk", "no": "Norwegian", "oc": "Occitan", "or": "Odia",
    "pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese", "rm": "Romansh", "ro": "Romanian",
    "ru": "Russian", "rw": "Kinyarwanda", "sa": "Sanskrit", "sc": "Sardinian", "sd": "Sindhi", "si": "Sinhala",
    "sk": "Slovak", "sl": "Slovenian", "sm": "Samoan", "sn": "Shona", "so": "Somali", "sq": "Albanian",
    "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
    "tg": "Tajik", "th": "Thai", "ti": "Tigrinya", "tk": "Turkmen", "tl": "Tagalog", "tr": "Turkish", "tt": "Tatar",
    "ug": "Uyghur", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese", "wo": "Wolof", "xh": "Xhosa",
    "yi": "Yiddish", "yo": "Yoruba", "yue": "Cantonese", "zh": "Chinese", "zu": "Zulu",
}
LEGACY = {"iw": "he", "in": "id", "ji": "yi"}


def lang_name(code: str) -> str:
    """English display name exactly as the app computes it. Pass a bare ISO 639 code ("pt", not "pt_BR": Java treats
    "pt_BR" as an unknown language and returns "pt_br")."""
    c = code.lower()
    c = LEGACY.get(c, c)
    if c in JAVA_ENGLISH_NAMES:
        return JAVA_ENGLISH_NAMES[c]
    try:   # CLDR via babel, if installed; Java/ICU use the same data
        from babel import Locale
        name = Locale.parse(c).get_language_name("en")
        if name:
            return name
    except Exception:  # noqa: BLE001  (babel missing or unknown code)
        pass
    return c


# ---------------------------------------------------------------- the app's prompt
TASK = {"rambler": "Rewrite as a short, clear message",
        "light": "Clean up: punctuation, capitals, remove fillers, apply self-corrections, keep the words"}


def system_prompt(mode: str, lang: str) -> str:
    return f"{TASK[mode]}. Language: {lang_name(lang)}. Never translate. Keep slang and swearing exactly. Output only the text."


def raw_prompt(mode: str, lang: str, text: str) -> str:
    """Qwen3 chat template with thinking disabled, as the app builds it by hand for the on-device model. Equal to
    tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False) for Qwen3."""
    return (f"<|im_start|>system\n{system_prompt(mode, lang)}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def chat_prompt(tok, mode: str, lang: str, text: str) -> str:
    msgs = [{"role": "system", "content": system_prompt(mode, lang)}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)


# ---------------------------------------------------------------- per-language config
class Lang:
    """languages/<code>.json: filler words, correction phrases, numbers, swear stems, hard test cases."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.code = cfg["code"]
        self.name = lang_name(self.code)
        self.fillers = cfg["fillers"]
        self.corrections = cfg["corrections"]
        self.trailing_corrections = cfg.get("trailing_corrections") or self.corrections
        self.light_fillers = cfg.get("light_fillers") or self.fillers[:3]
        self.numbers = cfg.get("numbers", [])
        self.swear_stems = tuple(strip_acc(s) for s in cfg.get("swear_stems", []))
        self.swear_exact = {strip_acc(s) for s in cfg.get("swear_exact", [])}
        self.detect = re.compile(cfg["detect"], re.I) if cfg.get("detect") else None
        self.hard_cases = cfg.get("hard_cases", [])
        m = {"filler": 0.10, "repeat": 0.04, "false_start": 0.02, "correction": 0.35, "lead_filler": 0.30,
             "correction_filler": 0.08}
        m.update(cfg.get("messy", {}))
        self.p = m


def load_lang(spec: str) -> Lang:
    p = pathlib.Path(spec)
    if p.suffix not in (".json", ".jsonc"):
        p = HERE / "languages" / f"{spec}.json"
    if not p.exists():
        raise SystemExit(f"no config {p}: copy languages/template.jsonc to languages/<code>.json and fill it in")
    text = "\n".join(l for l in p.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("//"))
    return Lang(json.loads(text))


def strip_acc(w: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", w.lower()) if unicodedata.category(c) != "Mn")


def swears(text: str, langs) -> int:
    """Swear words in text, by the stems of all given languages (people mix languages when they swear)."""
    stems = tuple(s for l in langs for s in l.swear_stems)
    exact = set().union(*(l.swear_exact for l in langs)) if langs else set()
    n = 0
    for w in re.findall(r"[\w']+", text):
        w = strip_acc(w)
        n += w in exact or bool(stems and w.startswith(stems))
    return n


def kept_language(output: str, target: str, lang: Lang) -> bool:
    """Language check for evaluation: the output looks like the language exactly when the target does (catches a
    translation into another language). Uses the config's `detect` regex (a script range or common words)."""
    if not lang.detect:
        return True
    return bool(lang.detect.search(output)) == bool(lang.detect.search(target))


# ---------------------------------------------------------------- messy dictation
def messy(text: str, lang: Lang, rng) -> str:
    """Turn a clean message into plausible raw dictation: lowercase, no punctuation, fillers, repeated words, false
    starts, an inline self-correction ("..., no wait, ...") and sometimes a leading filler."""
    p = lang.p
    words = re.findall(r"[\w'’-]+", text.lower())
    out = []
    for i, w in enumerate(words):
        r = rng.random()
        if r < p["filler"]:
            out.append(rng.choice(lang.fillers))                          # filler before a word
        elif r < p["filler"] + p["repeat"]:
            out.append(w)                                                 # repeated word ("the the")
        elif r < p["filler"] + p["repeat"] + p["false_start"] and i + 2 < len(words):
            out += [w, words[i + 1], rng.choice(lang.fillers)]            # false start: "I was um I was"
        out.append(w)
    if len(out) > 5 and rng.random() < p["correction"]:                  # say a wrong word, then fix it
        j = rng.randrange(1, len(out) - 1)
        wrong = rng.choice([x for x in out if len(x) > 3] or [out[j]])
        if wrong != out[j]:
            out = out[:j] + [wrong, rng.choice(lang.corrections)] + out[j:]
    if rng.random() < p["lead_filler"]:
        out.insert(0, rng.choice(lang.fillers))
    return " ".join(out)


# ---------------------------------------------------------------- data files
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(f, row):
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- models (evaluation)
def load_model(path: str, base: str = "Qwen/Qwen3-1.7B", device: str = "auto"):
    """A merged model folder / HF id, or a LoRA adapter folder (adapter_config.json) on top of `base`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    if (pathlib.Path(path) / "adapter_config.json").exists():
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(base)
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype), path)
    else:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    model = model.to(dev).eval()

    def run(mode, lang, text):
        ids = tok(chat_prompt(tok, mode, lang, text), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=int(len(text) * 0.8) + 40, do_sample=False)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    return run, model
