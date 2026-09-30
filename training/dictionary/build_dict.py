"""Build a FUTO Keyboard dictionary for any language: text corpora -> <code>_wordlist.combined -> main_<code>.dict.

Everything language-specific lives in languages/<code>.json (see languages/template.jsonc):
  - words and frequencies are counted in the configured corpora; each source has a boost, so informal registers
    (subtitles, chat, comments, slang lists) weigh more than their raw size and chat vocabulary ranks realistically
  - casing: a word is stored capitalised only if >= 90% of its occurrences are capitalised (names), else lowercase
  - frequency: f = 255 + log(p) / log(1.15), clipped to [min_f, 250]; AOSP treats f=0 as "never suggest", so no word
    that is kept ever gets 0
  - vulgar swearing is flagged possibly_offensive=true WITH its normal frequency: FUTO's "Block offensive words"
    setting then decides whether it is suggested, and it stays a valid word either way (never autocorrected away)
  - optionally, an existing AOSP wordlist is merged at low frequency, so valid rare forms are not "corrected"
Compiling needs Java and dicttool_aosp.jar (downloaded on first --compile, see README).

Usage: python build_dict.py --lang sq --data <corpus folder> [--out build] [--compile]
"""
import argparse
import datetime
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import pathlib
import re
import shutil
import subprocess
import unicodedata
import urllib.request
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
# Helium314's copy of the AOSP dicttool (built by remi0s/aosp-dictionary-tools from AOSP, Apache-2.0)
DICTTOOL_URL = "https://codeberg.org/Helium314/aosp-dictionaries/raw/branch/main/dicttool_aosp.jar"
DICTTOOL_SHA256 = "a8c5bd21f631ed0a92235d42d2fe83af5d70216172bf7e22781a9a946858237e"


def load_config(spec):
    """`sq` -> languages/sq.json; or a path. Full-line // comments are allowed (the template uses them)."""
    p = pathlib.Path(spec)
    if p.suffix not in (".json", ".jsonc"):
        p = HERE / "languages" / f"{spec}.json"
    if not p.exists():
        raise SystemExit(f"no config {p}: copy languages/template.jsonc to languages/<code>.json and fill it in")
    text = "\n".join(l for l in p.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("//"))
    cfg = json.loads(text)
    if "," in cfg["description"] or "=" in cfg["description"]:
        raise SystemExit("description must not contain ',' or '=' (the .combined header is key=value,key=value)")
    return cfg


def nfc(t):
    return unicodedata.normalize("NFC", t)


class Offensive:
    """possibly_offensive check. Matching is on the lowercased NFC form WITH accents, because swear words often have
    innocent look-alikes that differ only by accent or ending (Greek μαλάκα / μαλακά = softly, γαμώ / γάμος = wedding)."""

    def __init__(self, cfg):
        low = lambda xs: tuple(nfc(x.lower()) for x in xs)  # noqa: E731
        self.stems = low(cfg.get("offensive_stems", []))
        self.exact = set(low(cfg.get("offensive_exact", [])))
        self.not_stems = low(cfg.get("not_offensive_stems", []))
        self.not_exact = set(low(cfg.get("not_offensive_exact", [])))
        self.cap_stems = low(cfg.get("capitalised_offensive_stems", []))

    def __call__(self, w):
        lw = nfc(w.lower()).replace("’", "'")
        if w[:1].isupper() and not (self.cap_stems and lw.startswith(self.cap_stems)):
            return False   # stored capitalised only when (nearly) always capitalised: names and places, not swearing
        if (self.not_stems and lw.startswith(self.not_stems)) or lw in self.not_exact:
            return False
        return lw in self.exact or bool(self.stems and lw.startswith(self.stems))


def count_file(args):
    path, boost, field, word_re = args
    word = re.compile(word_re)
    c = Counter()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if field:                                   # jsonl: count one field (e.g. transcripts' "text")
                try:
                    line = json.loads(line).get(field) or ""
                except json.JSONDecodeError:
                    continue
            c.update(word.findall(nfc(line)))
    return {w: n * boost for w, n in c.items()}


def count_sources(cfg, data):
    jobs = []
    for s in cfg["sources"]:
        files = sorted(glob.glob(str(pathlib.Path(data) / s["path"])))
        if not files:
            print(f"warning: no files for source {s['path']!r} under {data}", flush=True)
        field = s.get("field") or ("text" if s["path"].endswith(".jsonl") else None)
        jobs += [(f, s.get("boost", 1), field, cfg["word_regex"]) for f in files]
    if not jobs:
        raise SystemExit("no corpus files found: check --data and the config's sources")
    total = Counter()
    with mp.Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        for part in pool.map(count_file, jobs):
            total.update(part)
    return total


def merge_case(cfg, total):
    """One form per word: "Tirana" only if >= capitalised_share of uses are capitalised, else the lowercase form."""
    share = cfg.get("capitalised_share", 0.9)
    skip = re.compile(cfg["skip_regex"]) if cfg.get("skip_regex") else None
    by_lower = {}
    for w, n in total.items():
        by_lower.setdefault(w.lower(), Counter())[w] += n
    words = {}
    for low, forms in by_lower.items():
        n = sum(forms.values())
        cap = sum(v for k, v in forms.items() if k[:1].isupper())
        form = max((k for k in forms if k[:1].isupper()), key=forms.get) if cap >= share * n and len(low) > 1 else low
        if form.isupper() and len(form) > 1:            # ALL-CAPS only: headlines and acronyms
            if cfg.get("all_caps", "titlecase") == "skip":
                continue
            form = form.capitalize()
        if skip and skip.search(low):                   # never a word (e.g. Greek final σ instead of ς: a typo)
            continue
        words[form] = n
    for w in cfg.get("core_words", []):                 # informal core vocabulary: always included
        w = nfc(w)
        words[w] = max(words.get(w, 0), 1)
    return words


def to_rows(cfg, words):
    core = {nfc(c.lower()) for c in cfg.get("core_words", [])}
    singles = cfg.get("single_letters", "")
    drop = re.compile(cfg["drop_regex"]) if cfg.get("drop_regex") else None
    min_n, min_cap = cfg.get("min_count", 30), cfg.get("min_count_capitalised", 600)
    min_f, core_f, max_f = cfg.get("min_f", 30), cfg.get("core_min_f", 150), cfg.get("max_f", 250)
    total_n = sum(words.values())
    rows = []
    for w, n in words.items():
        if len(w) > 30 or (len(w) == 1 and w.lower() not in singles):
            continue
        is_core = w.lower() in core
        if not is_core and (n < min_n or (w[:1].isupper() and n < min_cap) or (drop and drop.search(w.lower()))):
            continue   # one-offs and typos (boosted counts), rare capitalised web noise, unwanted forms
        f = int(round(255 + math.log(n / total_n) / math.log(1.15)))
        rows.append((min(max(f, core_f if is_core else min_f), max_f), w))
    return rows


def merge_extra(cfg, rows, cache):
    """Add an existing wordlist's forms we do not have, at low frequency (valid rare inflections)."""
    ex = cfg.get("extra_wordlist")
    if not ex:
        return rows
    path = pathlib.Path(cache) / pathlib.Path(ex["url"]).name
    if not path.exists():
        print(f"downloading {ex['url']}", flush=True)
        urllib.request.urlretrieve(ex["url"], path)
    word = re.compile(cfg["word_regex"])
    have = {w.lower() for _, w in rows}
    added = 0
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*word=([^,]+),\s*f=(\d+)", line)
        if not m:
            continue
        w = nfc(m.group(1))
        if w.lower() in have or not word.fullmatch(w) or len(w) == 1:
            continue
        rows.append((min(int(m.group(2)), ex.get("max_f", 100)), w)); have.add(w.lower()); added += 1
    print(f"added {added} forms from {path.name} at f <= {ex.get('max_f', 100)}", flush=True)
    return rows


def write_combined(cfg, rows, path, offensive):
    code = cfg["locale"]
    header = (f"dictionary=main:{code.lower()},locale={code},description={cfg['description']},"
              f"date={int(datetime.datetime.now().timestamp())},version={cfg.get('version', 1)}")
    with open(path, "w", encoding="utf-8") as out:
        out.write(header + "\n")
        for f, w in rows:
            out.write(f" word={w},f={f}" + (",possibly_offensive=true" if offensive(w) else "") + "\n")


def dicttool(path):
    p = pathlib.Path(path)
    if not p.exists():
        print(f"downloading dicttool_aosp.jar from {DICTTOOL_URL}", flush=True)
        urllib.request.urlretrieve(DICTTOOL_URL, p)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != DICTTOOL_SHA256:
            print(f"warning: dicttool_aosp.jar sha256 {digest} differs from the tested one; it may have been updated")
    return p


def compile_dict(combined, out_dict, jar):
    home = os.environ.get("JAVA_HOME")
    java = shutil.which("java", path=os.path.join(home, "bin")) if home else None
    java = java or shutil.which("java")
    if not java:
        raise SystemExit("java not found: install a JRE (11+) or compile later with\n"
                         f"  java -jar dicttool_aosp.jar makedict -s {combined} -d {out_dict}")
    subprocess.run([java, "-jar", str(jar), "makedict", "-s", str(combined), "-d", str(out_dict)], check=True)
    print(f"wrote {out_dict} ({out_dict.stat().st_size / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, help="language code (languages/<code>.json) or a config path")
    ap.add_argument("--data", default=".", help="folder the config's source paths are relative to")
    ap.add_argument("--out", default="build", help="output folder")
    ap.add_argument("--compile", action="store_true", help="also compile main_<code>.dict (needs Java)")
    ap.add_argument("--dicttool", default=str(HERE / "dicttool_aosp.jar"), help="path to dicttool_aosp.jar")
    a = ap.parse_args()

    cfg = load_config(a.lang)
    code = cfg["locale"]
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    offensive = Offensive(cfg)

    words = merge_case(cfg, count_sources(cfg, a.data))
    rows = merge_extra(cfg, to_rows(cfg, words), out)
    rows.sort(reverse=True)
    combined = out / f"{code}_wordlist.combined"
    write_combined(cfg, rows, combined, offensive)

    flagged = sorted({w for _, w in rows if offensive(w)})
    print(f"{len(rows)} words -> {combined}; top: {[w for _, w in rows[:15]]}")
    print(f"{len(flagged)} flagged possibly_offensive (review these!): {', '.join(flagged[:200])}"
          + (" ..." if len(flagged) > 200 else ""))
    fmap = {w: f for f, w in rows}
    print("spot check:", {w: fmap.get(nfc(w)) for w in cfg.get("look", [])})
    if a.compile:
        compile_dict(combined, out / f"main_{code}.dict", dicttool(a.dicttool))


if __name__ == "__main__":
    main()
