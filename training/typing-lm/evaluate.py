"""Evaluate a typing LM the way FUTO Keyboard uses it.

  next-word  : context -> top-3 next words (beam over pieces until a piece ends in "▁"), vs the true next word
  autocorrect: context <XBU> simulated taps <XBC> -> top-3 words, vs the intended word. Three typing conditions:
               clean   = accurate taps, no errors (tests diacritic restoration + noise tolerance)
               typos   = sloppy taps plus at least one keystroke error / misspelling (the training error model)
               partial = only the first ~60% of the letters (tests word completion)
Test text: held-out lines from the training sources (<tok>/*.val.txt) and the config's eval.text files.
eval.pairs (optional): type each MISSPELLED word with accurate taps, no context; is the correct word top-1 / top-3?

Usage: python evaluate.py <model dir> [--tok <tok dir>] [--n 150] [--no-pairs]
"""
import argparse
import csv
import io
import pathlib
import random
import re
import unicodedata
import urllib.request
from collections import defaultdict

import sentencepiece as spm
import torch
from transformers import LlamaForCausalLM

import langconfig

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--lang", default="", help="language code or config (default: the model dir's language.yaml)")
p.add_argument("--tok", default="", help="tok dir with *.val.txt (default: the training data dir)")
p.add_argument("--n", type=int, default=150, help="sentences per test set")
p.add_argument("--error-tags", default="", help="typing_errors tags for the 'typos' condition")
p.add_argument("--no-pairs", action="store_true")
p.add_argument("--device", default="", help="cuda / mps / cpu (default: best available)")
args = p.parse_args()
DEV = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

MODEL = pathlib.Path(args.model)
CFG = langconfig.load(args.lang or str(MODEL / "language.yaml"))
sp = spm.SentencePieceProcessor(model_file=str(MODEL / "tokenizer.model"))
L = langconfig.Language(CFG, [t for t in args.error_tags.split(",") if t]).bind(sp)
XBU, XBC = sp.piece_to_id("<XBU>"), sp.piece_to_id("<XBC>")
model = LlamaForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV).eval()
emb = model.get_input_embeddings()
ENDS = torch.tensor([sp.id_to_piece(i).endswith("▁") for i in range(sp.get_piece_size())])


def norm(w):
    return unicodedata.normalize("NFC", re.sub(r"[^\w]", "", w.lower()))


@torch.no_grad()
def top_words(rows_t, rows_w, k=3, beam=8, max_pieces=10):
    """rows: per position (4 token ids, 4 weights). Returns top-k words (beam until a piece ends in ▁).
    No KV cache: every step re-runs all beams in one batch (the model and sequences are tiny)."""
    t = torch.tensor(rows_t, device=DEV); w = torch.tensor(rows_w, dtype=torch.float32, device=DEV)
    prefix = (emb(t) * w[..., None]).sum(1)                      # (L, d)
    beams, done = [(0.0, [])], []
    for _ in range(max_pieces):
        if not beams:
            break
        n = max(len(b) for _, b in beams)  # right-padding is harmless: causal attention never looks ahead
        ext = torch.stack([torch.cat([prefix, emb(torch.tensor(b + [0] * (n - len(b)), dtype=torch.long, device=DEV))])
                           for _, b in beams])
        logits = model(inputs_embeds=ext).logits
        cand = []
        for (s, b), lg in zip(beams, logits):
            lp = torch.log_softmax(lg[prefix.shape[0] + len(b) - 1].float(), -1)
            v, i = lp.topk(beam)
            cand += [(s + vv.item(), b + [ii.item()]) for vv, ii in zip(v, i)]
        cand.sort(key=lambda c: -c[0])
        beams = []
        for s, b in cand:
            if ENDS[b[-1]]:
                wd = sp.decode(b).strip()
                if wd and norm(wd) not in [norm(d) for _, d in done]:
                    done.append((s, wd))
            elif len(beams) < beam:
                beams.append((s, b))
        if len(done) >= k and (not beams or done[k - 1][0] > beams[0][0]):
            break
    done.sort(key=lambda c: -c[0])
    return [d for _, d in done[:k]]


def plain(ids):
    return [[i, 0, 0, 0] for i in ids], [[1, 0, 0, 0] for _ in ids]


def taps(ctx, keys, sigma, rng):
    rt, rw = plain(ctx + [XBU])
    for k in keys:
        kt, kw = L.tap_mix(k, sigma, rng); rt.append(kt.tolist()); rw.append(kw.tolist())
    rt.append([XBC, 0, 0, 0]); rw.append([1, 0, 0, 0])
    return rt, rw


def run(sentences, name):
    rng = random.Random(42)
    res = {"next-word": [0, 0, 0], "clean": [0, 0, 0], "typos": [0, 0, 0], "partial": [0, 0, 0]}
    for s in sentences:
        words = s.split()
        for i, word in enumerate(words):
            ctx = [1] + sp.encode(" ".join(words[:i]) + " ") if i else [1]
            gold = norm(word)
            if i:  # next-word (before any key is typed)
                preds = [norm(p) for p in top_words(*plain(ctx))]
                r = res["next-word"]; r[0] += bool(preds) and preds[0] == gold; r[1] += gold in preds; r[2] += 1
            core = re.sub(r"[^\w]", "", word)
            if len(core) < 2 or not L.is_word(core):
                continue
            base = L.base_letters(core)
            for mode in ("clean", "typos", "partial"):
                if mode == "clean":
                    keys, sigma = list(base), 0.15
                elif mode == "typos":
                    keys, sigma = None, 0.35
                    for _ in range(20):  # force at least one keystroke-level error
                        keys = L.keystrokes(core, rng)
                        if len(keys) >= len(base) - 1 and keys != list(base):
                            break
                else:
                    keys, sigma = list(base[:max(1, round(len(base) * 0.6))]), 0.15
                keys = [k for k in keys if k in L.neighbours]
                if not keys:
                    continue
                preds = [norm(p) for p in top_words(*taps(ctx, keys, sigma, rng))]
                r = res[mode]; r[0] += bool(preds) and preds[0] == gold; r[1] += gold in preds; r[2] += 1
    for k, (t1, t3, n) in res.items():
        print(f"{name:22s} {k:10s} top-1 {t1 / max(1, n) * 100:5.1f}%  top-3 {t3 / max(1, n) * 100:5.1f}%  ({n})", flush=True)


def pairs(conf):
    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(conf["url"]).read().decode("utf-8"))))
    rng = random.Random(0)
    res = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        wrong, right = row[conf["wrong"]].strip().lower(), row[conf["right"]].strip().lower()
        keys = [c for c in wrong if c in L.neighbours]
        if not keys:
            continue
        preds = [norm(p) for p in top_words(*taps([1], keys, 0.12, rng))]
        for key in ("ALL", row.get(conf.get("type", ""), "")):
            if key:
                r = res[key]; r[0] += bool(preds) and preds[0] == norm(right); r[1] += norm(right) in preds; r[2] += 1
    for et, (t1, t3, n) in sorted(res.items(), key=lambda x: x[0] != "ALL"):
        print(f"pairs {et:22s} n={n:5d}  top-1 {t1 / n * 100:5.1f}%  top-3 {t3 / n * 100:5.1f}%", flush=True)


if __name__ == "__main__":
    import json
    tok = pathlib.Path(args.tok or json.load(open(MODEL / "train_args.json"))["data"])
    held = [l.strip() for f in sorted(tok.glob("*.val.txt")) for l in open(f, encoding="utf-8") if 20 <= len(l) <= 200]
    random.Random(0).shuffle(held)
    sets = {"held-out (sources)": held[:args.n]}
    for pattern in CFG["eval"].get("text", []):
        for path in sorted(pathlib.Path(CFG["_dir"]).glob(pattern)):
            sets[path.stem] = [l.strip() for l in open(path, encoding="utf-8") if l.strip()][:args.n]
    for name, sents in sets.items():
        run(sents, name)
    if CFG["eval"].get("pairs") and not args.no_pairs:
        pairs(CFG["eval"]["pairs"])
