"""Train a typing LM for FUTO Keyboard: next-word prediction + autocorrect, one small Llama.

Two kinds of examples, interleaved (each batch is ~half and half):
  LM  : <s> text...                                        loss on every next token (next-word prediction)
  AC  : <s> context▁ <XBU> k1 k2 ... kn <XBC> word pieces▁  loss only on the word pieces (autocorrect/completion)

The AC input is exactly what FUTO Keyboard builds in PredictCorrection (with patch 0001): BOS + tokenize(context + " ")
+ <XBU> + one input position per keystroke, whose embedding is the weighted mix of the embeddings of the 4 nearest
letter keys (char_embed_mixing_v1) + <XBC>; the model then writes the word, ending at the first piece that ends in "▁".
Taps are simulated on the layout in languages/<code>.yaml: long-press diacritics are typed as the base letter, so the
model learns to restore them. Errors: fuzzy taps, dropped / extra / swapped / doubled keys, the language's own
misspellings (typing_errors), and partial words (the first letters only -> word completion).

Default model: 8 layers x 512, FFN 1536, 16k vocab = 35.5M parameters (GGUF: 84 MB F16, 35 MB Q6_K).

Usage: python train.py --lang xx --data work/xx/tok --out work/xx/model [--tokens 1.5e9] [--probe]
"""
import argparse
import json
import math
import pathlib
import random
import shutil
import time

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import LlamaConfig, LlamaForCausalLM

import langconfig

# parsed at import: DataLoader workers re-import this module (spawn on Windows/macOS) and need the same settings
p = argparse.ArgumentParser()
p.add_argument("--lang", required=True, help="language code or config path")
p.add_argument("--data", required=True, help="tok dir from prepare.py")
p.add_argument("--out", required=True)
p.add_argument("--tokens", type=float, default=1.5e9, help="total training tokens (LM + AC positions)")
p.add_argument("--seq", type=int, default=128)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=1.5e-3)
p.add_argument("--layers", type=int, default=8)
p.add_argument("--hidden", type=int, default=512)
p.add_argument("--ffn", type=int, default=1536)       # multiple of 256: Q6_K-friendly, even 32-blocks
p.add_argument("--ac_frac", type=float, default=0.5)  # share of AC examples
p.add_argument("--workers", type=int, default=12)
p.add_argument("--probe", action="store_true", help="a few steps + speed report, then exit")
p.add_argument("--init", default="", help="continue from this model dir (fine-tune) instead of random init")
p.add_argument("--error-tags", default="", help="comma-separated typing_errors tags to enable (e.g. v2)")
p.add_argument("--save-every", type=int, default=5000)
p.add_argument("--device", default="", help="cuda / mps / cpu (default: best available)")
args = p.parse_args()
DEV = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

CFG = langconfig.load(args.lang)
DATA, OUT = pathlib.Path(args.data), pathlib.Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
sp = spm.SentencePieceProcessor(model_file=str(DATA / "tokenizer.model"))
V = sp.get_piece_size()
BOS, PAD = 1, 3
XBU, XBC, XEC = (sp.piece_to_id(t) for t in ("<XBU>", "<XBC>", "<XEC>"))
assert min(XBU, XBC, XEC) > 3, "control tokens missing from tokenizer"
SPACE = sp.piece_to_id("▁")  # plain space piece: FUTO's word end after banned punctuation
LANG = langconfig.Language(CFG, [t for t in args.error_tags.split(",") if t]).bind(sp)
# sampling weight per source (LM and AC); small sources capped so they repeat ~15-75x, not hundreds
MIX = {k: float(s.get("mix", 0)) for k, s in CFG["sources"].items() if s.get("mix", 0) > 0}


# ---------------------------------------------------------------- data
class Source:
    def __init__(self, name):
        self.ids = np.memmap(DATA / f"{name}.train.bin", dtype=np.uint16, mode="r")
        self.idx = np.fromfile(DATA / f"{name}.train.idx", dtype=np.uint64)
        self.n = len(self.idx) - 1

    def line(self, i):
        return self.ids[int(self.idx[i]):int(self.idx[i + 1])].astype(np.int64)


class Examples(IterableDataset):
    def __iter__(self):
        wi = get_worker_info()
        rng = random.Random((wi.id if wi else 0) * 1000003 + int(time.time()))
        srcs = {k: Source(k) for k in MIX if (DATA / f"{k}.train.bin").exists()}
        srcs = {k: s for k, s in srcs.items() if s.n > 0}
        names, weights = list(srcs), [MIX[k] for k in srcs]
        ends_word = np.array([sp.id_to_piece(i).endswith("▁") for i in range(V)])
        while True:
            src = srcs[rng.choices(names, weights)[0]]
            if rng.random() < args.ac_frac:
                ex = self.ac(src, rng, ends_word)
                if ex is not None:
                    yield ex
            else:
                yield self.lm(src, rng)

    def lm(self, src, rng):
        toks = [BOS]
        i = rng.randrange(src.n)
        while len(toks) < args.seq + 1:
            toks.extend(src.line(i).tolist()); i = (i + 1) % src.n
        toks = toks[:args.seq + 1]
        ids = np.array(toks[:-1]); labels = np.array(toks[1:])
        mix_t = np.zeros((args.seq, 4), np.int64); mix_t[:, 0] = ids
        mix_w = np.zeros((args.seq, 4), np.float32); mix_w[:, 0] = 1
        return mix_t, mix_w, labels

    def ac(self, src, rng, ends_word):
        line = src.line(rng.randrange(src.n))
        bounds = [0] + [j + 1 for j in np.flatnonzero(ends_word[line])]
        words = [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]
        # A word followed by punctuation ("word?") is pieces "word" + "?▁". FUTO bans punctuation pieces while
        # suggesting and moves their probability to the space token, so the target is the letters + "▁".
        cand = []
        for a, b in words:
            e = b
            while e > a and not sp.id_to_piece(int(line[e - 1]))[:1].isalpha():
                e -= 1  # drop trailing punctuation-only pieces
            w = sp.decode(line[a:e].tolist()).strip()
            if 2 <= len(w) <= 24 and LANG.is_word(w):
                cand.append((a, e, b))
        if not cand:
            return None
        a, e, b = rng.choice(cand)
        word = sp.decode(line[a:e].tolist()).strip()
        keys = LANG.keystrokes(word, rng)
        target = line[a:e].tolist() + ([] if e == b else [SPACE])
        if not keys or len(target) > 10:
            return None
        ctx = [BOS] + line[max(0, a - 60):a].tolist()
        sigma = rng.uniform(0.12, 0.40)
        t_rows, w_rows = [[t, 0, 0, 0] for t in ctx + [XBU]], [[1, 0, 0, 0] for _ in ctx + [XBU]]
        for k in keys:
            t, w = LANG.tap_mix(k, sigma, rng); t_rows.append(t.tolist()); w_rows.append(w.tolist())
        seq_t = t_rows + [[XBC, 0, 0, 0]] + [[t, 0, 0, 0] for t in target]
        seq_w = w_rows + [[1, 0, 0, 0]] * (1 + len(target))
        # label at position i = token that must follow input i: <XBC> -> first piece, ..., last piece -> nothing
        labels = [-100] * len(t_rows) + target + [-100]
        assert len(labels) == len(seq_t)
        seq_t, seq_w, labels = seq_t[:args.seq], seq_w[:args.seq], labels[:args.seq]
        pad = args.seq - len(seq_t)
        mix_t = np.array(seq_t + [[PAD, 0, 0, 0]] * pad, np.int64)
        mix_w = np.array(seq_w + [[1, 0, 0, 0]] * pad, np.float32)
        lab = np.array(labels + [-100] * (args.seq - len(labels)), np.int64)
        return mix_t, mix_w, lab


def collate(batch):
    return tuple(torch.from_numpy(np.stack(x)) for x in zip(*batch))


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()
    elif DEV == "mps":
        torch.mps.synchronize()


def main():
    assert MIX, "no source has a mix weight > 0"
    cfg = LlamaConfig(vocab_size=V, hidden_size=args.hidden, intermediate_size=args.ffn, num_hidden_layers=args.layers,
                      num_attention_heads=args.hidden // 64, num_key_value_heads=args.hidden // 64,
                      max_position_embeddings=256, rms_norm_eps=1e-5, tie_word_embeddings=True,
                      bos_token_id=BOS, eos_token_id=2, pad_token_id=PAD)
    model = (LlamaForCausalLM.from_pretrained(args.init) if args.init else LlamaForCausalLM(cfg)).to(DEV)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  vocab {V}  device {DEV}", flush=True)
    emb = model.get_input_embeddings()

    steps = max(1, int(args.tokens / (args.batch * args.seq)))
    warm = max(1, min(2000, steps // 20))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=DEV == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / steps)))))
    loader = DataLoader(Examples(), batch_size=args.batch, num_workers=args.workers, collate_fn=collate,
                        persistent_workers=args.workers > 0, prefetch_factor=8 if args.workers else None,
                        multiprocessing_context="spawn" if DEV == "mps" and args.workers else None)

    def forward(mix_t, mix_w, labels):
        x = (emb(mix_t.to(DEV, non_blocking=True)) * mix_w.to(DEV, non_blocking=True)[..., None]).sum(2)
        logits = model(inputs_embeds=x).logits
        return torch.nn.functional.cross_entropy(logits.float().view(-1, V), labels.to(DEV, non_blocking=True).view(-1),
                                                 ignore_index=-100)

    def save():
        model.save_pretrained(OUT)
        shutil.copy(DATA / "tokenizer.model", OUT / "tokenizer.model")
        # export/evaluate read the language from here; config_dir keeps its relative paths (extra/...) working
        (OUT / "language.yaml").write_text(pathlib.Path(CFG["_path"]).read_text(encoding="utf-8")
                                           + f"\nconfig_dir: '{CFG['_dir']}'\n", encoding="utf-8")
        json.dump(vars(args), open(OUT / "train_args.json", "w"), indent=1)

    print(f"{steps} steps x {args.batch} x {args.seq} = {steps * args.batch * args.seq / 1e9:.3f}B token positions", flush=True)
    model.train()
    log_every = min(200, max(1, steps // 10))
    t0, run = time.time(), 0.0
    for step, (mix_t, mix_w, labels) in enumerate(loader):
        if step >= steps:
            break
        with torch.autocast(DEV, dtype=torch.bfloat16):
            loss = forward(mix_t, mix_w, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run += loss.item()
        if args.probe and step == 10:
            sync(); t_probe = time.time()
        if args.probe and step == 60:
            sync(); el = time.time() - t_probe
            mem = f"  peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if DEV == "cuda" else ""
            print(f"PROBE {50 * args.batch * args.seq / el / 1e3:.0f}k tok/s{mem}  -> full run ~{steps * el / 50 / 3600:.1f} h",
                  flush=True)
            return
        if (step + 1) % log_every == 0:
            el = time.time() - t0
            print(f"step {step + 1}/{steps} loss {run / log_every:.3f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(step + 1) * args.batch * args.seq / el / 1e3:.0f}k tok/s eta {(steps - step - 1) * el / (step + 1) / 60:.0f} min",
                  flush=True)
            run = 0.0
        if (step + 1) % args.save_every == 0:
            save(); print(f"saved {OUT} at step {step + 1}", flush=True)
    save()
    print("done", flush=True)


if __name__ == "__main__":
    main()
