"""Pick source messages for the cleanup training data: short informal texts (6-60 words, 1-3 sentences), one per line.

The messages are the CLEAN side of every training pair (gen_pairs.py turns them into messy dictation), so they should
look like things people dictate to send: chat, dialogue, comments. Weight informal sources up; subtitles are ideal
(consecutive lines are joined into short exchanges).

Two ways to get them (combinable):
  --source FILE:SHARE[:join]  sample SHARE of --n from a local text file (one text per line); ":join" joins 1-3
                              consecutive lines (dialogue). --always FILE adds every line (e.g. slang sentences).
  --opus                      stream OPUS OpenSubtitles (monolingual) for the language and oversample lines with
                              swearing (by the config's swear stems), so the model sees enough of it.
Usage: python sample_sources.py --lang el --out work/sources_el.txt --n 8000 --source subs.txt:0.45:join \
         --source reddit.txt:0.15 --always slang.txt
       python sample_sources.py --lang en --out work/sources_en.txt --n 8000 --opus
--tidy gives every message a capital first letter and closing punctuation (for tweets and chat, which often lack
them: the light targets are the messages themselves).
"""
import argparse
import gzip
import random
import re
import unicodedata
import urllib.request

from common import load_lang, strip_acc

OPUS = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/{code}.txt.gz"


def ok(t):
    w = len(t.split())
    return 6 <= w <= 60 and not re.search(r"https?://|@\w|#\w|\d{4,}", t)


def sample(path, n, rng, join_lines):
    # reservoir sampling (uniform over the whole file), joining 1-3 consecutive lines for dialogue sources. The first
    # version replaced with a fixed ~2% chance, which over-weighted the start of the file (or its end, for big files).
    out, buf, seen = [], [], 0
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line:
            continue
        buf.append(line)
        k = rng.choice((1, 2, 3)) if join_lines else 1
        if len(buf) >= k:
            t = " ".join(buf); buf = []
            if ok(t):
                seen += 1
                if len(out) < n:
                    out.append(t)
                elif rng.random() < n / seen:
                    out[rng.randrange(n)] = t
    if len(out) < n:
        print(f"note: {path} has only {len(out)} usable messages (asked for {n})", flush=True)
    return out


def tidy(t):
    """Capital first letter and closing punctuation, so the light targets always look like finished messages (chat
    and tweet sources often have neither, and the model would learn to leave dictation unpunctuated)."""
    t = t.strip()
    i = next((i for i, c in enumerate(t) if c.isalpha()), None)
    if i is not None:
        t = t[:i] + t[i].upper() + t[i + 1:]
    if not t.endswith((".", "!", "?", "…", ")", '"', "»")):
        t += "."
    return t


def opus(lang, n, swear_share, rng, max_lines):
    """Stream the start of OpenSubtitles; keep ~10x more candidates than needed, then pick at random."""
    stems = lang.swear_stems
    want_sw, want_plain = int(n * swear_share), n - int(n * swear_share)
    plain, swear, buf = [], [], []
    with urllib.request.urlopen(OPUS.format(code=lang.code)) as r, gzip.GzipFile(fileobj=r) as gz:
        for i, raw in enumerate(gz):
            line = re.sub(r"<[^>]+>|^-\s*", "", raw.decode("utf-8", "replace")).strip()
            if not line or re.search(r"[♪#]|\d{3,}", line):
                continue
            buf.append(line)
            if len(buf) >= rng.choice((1, 2, 3)):
                t = " ".join(buf); buf = []
                if 6 <= len(t.split()) <= 60:
                    has = stems and any(strip_acc(w).startswith(stems) for w in re.findall(r"[\w']+", t))
                    (swear if has else plain).append(t)
            if (len(plain) >= 10 * want_plain and len(swear) >= 1.6 * want_sw) or i >= max_lines:
                break
    rng.shuffle(plain); rng.shuffle(swear)
    out = swear[:want_sw]
    out += plain[:n - len(out)]                      # too little swearing in the stream: fill with plain lines
    print(f"OPUS {lang.code}: {len(out)} messages, {min(len(swear), want_sw)} with swearing", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8000, help="messages to sample (before --always lines)")
    ap.add_argument("--source", action="append", default=[], help="FILE:SHARE[:join]")
    ap.add_argument("--always", action="append", default=[], help="FILE whose every line is included")
    ap.add_argument("--opus", action="store_true", help="sample --n messages from OPUS OpenSubtitles instead")
    ap.add_argument("--swear-share", type=float, default=0.3, help="share of OPUS messages that contain swearing")
    ap.add_argument("--max-lines", type=int, default=20_000_000, help="stop streaming OPUS after this many lines")
    ap.add_argument("--tidy", action="store_true",
                    help="capitalize the first letter and add a final period where missing (chat/tweet sources)")
    a = ap.parse_args()

    lang = load_lang(a.lang)
    rng = random.Random(lang.code)
    msgs = []
    for f in a.always:
        msgs += [l.strip() for l in open(f, encoding="utf-8") if l.strip()]
    for spec in a.source:
        join = spec.endswith(":join")
        path, share = (spec[:-5] if join else spec).rsplit(":", 1)   # rsplit: Windows paths contain ':'
        msgs += sample(path, int(a.n * float(share)), rng, join)
    if a.opus:
        msgs += opus(lang, a.n, a.swear_share, rng, a.max_lines)
    if not msgs:
        raise SystemExit("nothing sampled: give --source, --always or --opus")
    msgs = [unicodedata.normalize("NFC", m) for m in msgs]   # scraped text often has decomposed accents
    if a.tidy:
        msgs = [tidy(m) for m in msgs]
    msgs = list(dict.fromkeys(msgs))   # the same message from two sources
    rng.shuffle(msgs)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(msgs) + "\n")
    print(lang.code, len(msgs), "messages; e.g.:", *msgs[:3], sep="\n  ")


if __name__ == "__main__":
    main()
