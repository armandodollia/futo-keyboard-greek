"""Extra training pairs for TRAILING self-corrections, which a model trained only on gen_pairs.py data gets wrong:
   "... the little one is sick, sorry I meant the girl"   ->   the original sentence with the right words
Built from existing pairs: take a clean message m, swap one word (sometimes with the word before it) for a wrong one,
dictate it messily, then append "<correction phrase> <right words>", at the end or a bit later with the rest of the
message after it. Targets: light = m (exact), rambler = the teacher's existing rewrite of m. Numbers get their own
variant ("five, no wait, six") from the config's number words.
Usage: python augment_corrections.py --work work [--per-lang 3000]  ->  work/pairs_corrections.jsonl
"""
import argparse
import glob
import json
import os
import random
import re

from common import load_lang, write_jsonl


class Corrector:
    def __init__(self, langs, vocab, rng):
        self.langs, self.vocab, self.rng = langs, vocab, rng

    def dictate(self, words, lang):
        out = []
        for w in words:
            if self.rng.random() < lang.p["correction_filler"]:
                out.append(self.rng.choice(lang.light_fillers))
            out.append(w)
        return " ".join(out)

    def make(self, m, code):
        """Messy dictation of m with a wrong word that is corrected later on, or None if m is unsuitable."""
        lang, rng = self.langs[code], self.rng
        words = re.findall(r"[\w'’-]+", m.lower())
        if len(words) < 5:
            return None
        nums = [i for i, w in enumerate(words) if w in lang.numbers]
        if nums and rng.random() < 0.5:                      # number correction
            i = rng.choice(nums)
            wrong = rng.choice([n for n in lang.numbers if n != words[i]])
            right = [words[i]]
        else:
            cand = [i for i, w in enumerate(words) if len(w) >= 4 and i > 0]
            if not cand:
                return None
            i = rng.choice(cand)
            wrong = rng.choice(self.vocab[code])
            if wrong == words[i]:
                return None
            right = words[max(0, i - 1):i + 1] if rng.random() < 0.5 else [words[i]]   # sometimes repeat the article
        spoken = words[:i] + [wrong] + words[i + 1:]
        cut = rng.randrange(i + 1, len(spoken) + 1)            # the speaker notices a bit later, or at the very end
        tail = spoken[cut:]
        said = self.dictate(spoken[:cut], lang) + " " + rng.choice(lang.trailing_corrections) + " " + " ".join(right)
        if tail:   # ...and carries on. (The first version dropped the tail 30% of the time while the target kept it,
            said += " " + self.dictate(tail, lang)             # which taught the model to invent text.)
        return said


def load_messages(work):
    """(lang -> [(clean message, teacher rewrite or None)]) from the gen_pairs.py outputs."""
    light, ram = {}, {}
    for f in sorted(glob.glob(os.path.join(work, "pairs_*.jsonl"))):
        if "corrections" in f:
            continue
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            (light if r["mode"] == "light" else ram)[(r["lang"], r["input"])] = r["output"]
    msgs = {}
    for (code, inp), out in light.items():
        msgs.setdefault(code, []).append((out, ram.get((code, inp))))
    return msgs


def corrector_for(msgs, seed):
    langs = {code: load_lang(code) for code in msgs}
    vocab = {code: [w for m, _ in ms for w in re.findall(r"[^\W\d_]{4,}", m.lower())] for code, ms in msgs.items()}
    return Corrector(langs, vocab, random.Random(seed))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work", help="folder with pairs_*.jsonl")
    ap.add_argument("--per-lang", type=int, default=3000, help="correction items per language")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    msgs = load_messages(a.work)
    if not msgs:
        raise SystemExit(f"no pairs_*.jsonl in {a.work}: run gen_pairs.py first")
    c = corrector_for(msgs, a.seed)
    path = os.path.join(a.work, "pairs_corrections.jsonl")
    n = 0
    with open(path, "w", encoding="utf-8") as out:
        for code, ms in msgs.items():
            c.rng.shuffle(ms)
            k = 0
            for m, r in ms:
                s = c.make(m, code)
                if not s:
                    continue
                write_jsonl(out, {"lang": code, "mode": "light", "input": s, "output": m})
                if r:
                    write_jsonl(out, {"lang": code, "mode": "rambler", "input": s, "output": r})
                k += 1; n += 1
                if k >= a.per_lang:
                    break
    print("correction items:", n)
    for l in list(open(path, encoding="utf-8"))[:6]:
        r = json.loads(l); print(f"[{r['lang']}/{r['mode']}] {r['input'][:110]}\n   -> {r['output'][:110]}")


if __name__ == "__main__":
    main()
