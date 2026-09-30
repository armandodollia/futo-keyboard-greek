"""Shared helpers: language config, text normalisation, manifests, audio I/O, whisper.cpp tool lookup.

A language is described by a JSON file in languages/ (fields: README.md, "Language config"). `--lang xx` looks for
languages/xx.json; without one, any language Whisper knows works with plain defaults (lowercase, NFC, no
punctuation). `--lang path/to/file.json` uses that file.
"""
import json
import os
import pathlib
import re
import shutil
import unicodedata
import wave

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SR = 16000
SIZES = {"tiny": "39", "base": "74", "small": "244"}   # FUTO names its voice models by parameter count


# ---------------------------------------------------------------- language config
class Lang:
    def __init__(self, cfg: dict):
        from transformers.models.whisper.tokenization_whisper import LANGUAGES
        self.cfg = cfg
        self.code = cfg["code"]
        if self.code not in LANGUAGES:
            raise SystemExit(f"'{self.code}' is not a Whisper language code; FUTO and whisper.cpp can only force languages "
                             f"Whisper was trained on: {', '.join(sorted(LANGUAGES))}")
        self.whisper_language = cfg.get("whisper_language") or LANGUAGES[self.code]
        self.name = cfg.get("name") or self.whisper_language.title()
        self.fleurs = cfg.get("fleurs")
        self.teacher = cfg.get("teacher")
        n = cfg.get("normalize", {})
        self._form = n.get("unicode", "NFC")
        self._lower = n.get("lowercase", True)
        self._replace = n.get("replace", {})
        self._strip_marks = n.get("strip_marks", False)
        lo = cfg.get("loose", {})
        self._loose_replace = lo.get("replace", {})
        self._loose_strip_marks = lo.get("strip_marks", False)
        self._clean = [(re.compile(p), r) for p, r in cfg.get("clean", [])]

    def clean(self, t: str) -> str:
        """Fix a training transcript (kept cased and punctuated; only the config's `clean` rules are applied)."""
        t = unicodedata.normalize("NFC", t)
        for p, r in self._clean:
            t = p.sub(r, t)
        return re.sub(r"\s+", " ", t).strip()

    def norm(self, t: str) -> str:
        """Scoring form: WER/CER compare words, not casing or punctuation."""
        t = unicodedata.normalize(self._form, t.lower() if self._lower else t)
        for a, b in self._replace.items():
            t = t.replace(a, b)
        if self._strip_marks:
            t = _strip_marks(t)
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()

    def loose(self, t: str) -> str:
        """Tolerant form: also forgives diacritics that people (and subtitles) often drop, e.g. Albanian ë/ç, Greek tonos.
        Used for teacher checks of noisy labels and for the 'loose CER' column."""
        t = self.norm(t)
        for a, b in self._loose_replace.items():
            t = t.replace(a, b)
        return _strip_marks(t) if self._loose_strip_marks else t


def _strip_marks(t):
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn"))


def load_lang(spec: str) -> Lang:
    p = pathlib.Path(spec)
    if p.suffix != ".json":
        p = HERE / "languages" / f"{spec}.json"
    if p.exists():
        cfg = json.loads(p.read_text(encoding="utf-8"))
    elif spec.endswith(".json"):
        raise SystemExit(f"language config not found: {spec}")
    else:
        cfg = {"code": spec}
        print(f"note: no languages/{spec}.json, using default normalisation (see README.md, 'Language config')", flush=True)
    return Lang(cfg)


def cer(ref: str, hyp: str) -> float:
    import jiwer
    return jiwer.cer(ref, hyp) if ref else 1.0


def repeated(text: str) -> bool:
    """A runaway repetition: the decoder loops ("και και και ...", "> > > ..."). ACFT exists mostly to stop this in FUTO
    mode. Checked on the raw transcript: a loop of punctuation vanishes after normalisation but is still garbage."""
    w = text.lower().split()
    return len(w) > 8 and len(set(w)) < len(w) * 0.4


# ---------------------------------------------------------------- manifests (jsonl, one clip per line)
def read_jsonl(path) -> list:
    """Rows keep their fields; a relative "audio" path is resolved against the manifest's folder."""
    path = pathlib.Path(path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if "audio" not in r or "text" not in r:
                raise SystemExit(f"{path}:{n}: every row needs 'audio' and 'text'")
            a = pathlib.Path(r["audio"])
            r["audio"] = str(a if a.is_absolute() else (path.parent / a).resolve())
            rows.append(r)
    return rows


def write_jsonl(path, rows, quiet=False):
    """Audio paths are stored relative to the manifest when possible, so a data folder can be moved as a whole."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            r = dict(r)
            try:
                r["audio"] = pathlib.Path(os.path.relpath(r["audio"], path.parent.resolve())).as_posix()
            except ValueError:          # different drive on Windows
                pass
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not quiet:
        print(f"{path.name}: {len(rows)} clips, {sum(r.get('dur', 0) for r in rows) / 3600:.1f} h", flush=True)


# ---------------------------------------------------------------- audio
def load_audio(path) -> np.ndarray:
    """Any format ffmpeg/PyAV reads -> mono float32 at 16 kHz."""
    import av
    with av.open(str(path)) as c:
        res = av.AudioResampler(format="s16", layout="mono", rate=SR)
        chunks = [f.to_ndarray().reshape(-1) for fr in c.decode(audio=0) for f in res.resample(fr)]
        chunks += [f.to_ndarray().reshape(-1) for f in res.resample(None)]
    return (np.concatenate(chunks).astype(np.float32) / 32768.0) if chunks else np.zeros(SR // 10, np.float32)


def write_wav(path, audio: np.ndarray):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


# ---------------------------------------------------------------- devices and tools
def pick_device(want: str = "auto") -> str:
    import torch
    if want != "auto":
        return want
    return "cuda" if torch.cuda.is_available() else "cpu"


def find_tool(name: str, whisper_cpp: str = "") -> str:
    """whisper.cpp binaries (whisper-cli, whisper-quantize): --whisper-cpp <checkout or bin dir>, $WHISPER_CPP, or PATH."""
    exe = name + (".exe" if os.name == "nt" else "")
    for root in filter(None, [whisper_cpp, os.environ.get("WHISPER_CPP", "")]):
        root = pathlib.Path(root)
        for sub in ("", "bin", "build/bin", "build/bin/Release", "build/bin/Debug"):
            if (root / sub / exe).exists():
                return str(root / sub / exe)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} not found: build whisper.cpp (see README) and pass --whisper-cpp <its folder>, "
                         f"or set WHISPER_CPP")
    return found
