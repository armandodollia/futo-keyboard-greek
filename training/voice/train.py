"""Fine-tune Whisper (tiny / base / small) on one language, with optional knowledge distillation from a teacher.

Data: every <data>/train_*.jsonl (or --train files). Checkpoint selection: mean WER over the dev sets
(<data>/eval_*_dev.jsonl, up to --dev-clips each), evaluated every --eval-every steps; the best checkpoint is saved to
--out. At the end every <data>/eval_*_test.jsonl is scored with the best checkpoint (full 30 s context, not FUTO mode:
use evaluate.py on the .bin for that). Summary: <out>/train_info.json.

Distillation (--teacher, optional): the student also matches the teacher's token probabilities (Distil-Whisper recipe:
T = 2, loss = 0.8 x CE + 1.0 x KL). It helps most when the student is much smaller than the teacher, e.g. base/tiny
learning from your fine-tuned small, or small learning from a strong large-v3 fine-tune for your language. The teacher
may have a different vocabulary (large-v3 added <|yue|>) or 128 mel bins: tokens are matched by string, and the
teacher gets its own features. It costs one teacher forward pass per step and its memory.

Recipe defaults are the ones that worked for Greek and Albanian: AdamW, warmup then linear decay to 2%, weight decay
0.01, grad clip 1.0, SpecAugment masks 0.05, bf16 autocast on CUDA.
"""
import argparse
import json
import pathlib
import random
import sys
import time
from collections import Counter, defaultdict

import jiwer
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import SR, load_audio, load_lang, pick_device, read_jsonl

KD_T, KD_CE, KD_KL = 2.0, 0.8, 1.0


class ClipDataset(Dataset):
    def __init__(self, rows, fe, tok, teacher_fe=None):
        self.rows, self.fe, self.tok, self.tfe = rows, fe, tok, teacher_fe

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        a = load_audio(r["audio"])[: SR * 30]
        feats = self.fe(a, sampling_rate=SR, return_tensors="np").input_features[0]
        tfeats = self.tfe(a, sampling_rate=SR, return_tensors="np").input_features[0] if self.tfe else None
        return feats, tfeats, self.tok(r["text"]).input_ids


def collate(batch):
    feats = torch.tensor(np.stack([f for f, _, _ in batch]))
    tfeats = torch.tensor(np.stack([t for _, t, _ in batch])) if batch[0][1] is not None else None
    labels = torch.full((len(batch), max(len(l) for _, _, l in batch)), -100, dtype=torch.long)
    for i, (_, _, l) in enumerate(batch):
        labels[i, :len(l)] = torch.tensor(l)
    return feats, tfeats, labels


ALIASES = {"<|nocaptions|>": "<|nospeech|>"}   # renamed in large-v3


def token_map(student_tok, teacher_tok, n_student):
    """Student id -> teacher id, matched by token string (e.g. every id >= 50358 is +1 in large-v3)."""
    tv = teacher_tok.get_vocab()
    toks = student_tok.convert_ids_to_tokens(list(range(n_student)))
    ids = [tv.get(t, tv.get(ALIASES.get(t, ""))) for t in toks]
    missing = [t for t, i in zip(toks, ids) if i is None]
    if missing:
        print(f"warning: {len(missing)} student tokens not in the teacher's vocabulary (e.g. {missing[:3]}); "
              "they get the teacher's end-of-text probabilities", flush=True)
    return torch.tensor([teacher_tok.eos_token_id if i is None else i for i in ids], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--data", help="manifest folder (default work/<code>/data)")
    ap.add_argument("--train", nargs="*", help="training manifests (default: <data>/train_*.jsonl)")
    ap.add_argument("--base", default="openai/whisper-small", help="starting model: openai/whisper-{tiny,base,small} or a folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=1.25e-5, help="small 1.25e-5, base 2.5e-5, tiny 3.75e-5 worked well")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation: effective batch = batch x accum")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--mask", type=float, default=0.05, help="SpecAugment time/feature mask probability")
    ap.add_argument("--eval-every", type=int, default=600)
    ap.add_argument("--dev-clips", type=int, default=500, help="clips per dev set used for checkpoint selection")
    ap.add_argument("--cap-hours-per-group", type=float, default=0,
                    help="cap rows sharing a 'group' value (e.g. one audiobook) so a few voices don't dominate")
    ap.add_argument("--teacher", default="", help="optional KD teacher (Hugging Face id or folder)")
    ap.add_argument("--no-ckpt", action="store_true", help="disable gradient checkpointing (faster, needs more memory)")
    ap.add_argument("--probe", action="store_true", help="one worst-case step, print peak GPU memory, exit")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=-1, help="data loader workers (default 6 on CUDA, 0 on CPU)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke tests)")
    a = ap.parse_args()

    from transformers import (GenerationConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration,
                              WhisperProcessor, WhisperTokenizer)
    random.seed(0); torch.manual_seed(0)
    lang = load_lang(a.lang)
    dev = pick_device(a.device)
    cuda = dev == "cuda"
    data = pathlib.Path(a.data or f"work/{lang.code}/data")
    out = pathlib.Path(a.out)

    processor = WhisperProcessor.from_pretrained(a.base)
    processor.tokenizer.set_prefix_tokens(language=lang.whisper_language, task="transcribe")
    tok, fe = processor.tokenizer, processor.feature_extractor

    files = a.train or sorted(str(p) for p in data.glob("train_*.jsonl"))
    items, per_group = [], defaultdict(float)
    for f in files:
        rows = read_jsonl(f)
        random.shuffle(rows)
        for r in rows:
            if a.cap_hours_per_group and "group" in r:
                if per_group[r["group"]] + r.get("dur", 0) > a.cap_hours_per_group * 3600:
                    continue
                per_group[r["group"]] += r.get("dur", 0)
            items.append(dict(r, src=r.get("src", pathlib.Path(f).stem)))
    if not items:
        sys.exit(f"no training rows (looked for {data}/train_*.jsonl)")
    hours = sum(r.get("dur", 0) for r in items) / 3600
    print(f"training clips: {dict(Counter(r['src'] for r in items))}, {hours:.1f} h", flush=True)

    def eval_set(rows, model, bs=24):
        model.eval(); model.config.use_cache = True
        hyps = []
        for i in range(0, len(rows), bs):
            chunk = rows[i:i + bs]
            feats = fe([load_audio(r["audio"])[: SR * 30] for r in chunk], sampling_rate=SR, return_tensors="pt").input_features
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                ids = model.generate(feats.to(dev), language=lang.whisper_language, task="transcribe", max_new_tokens=220)
            hyps += processor.batch_decode(ids, skip_special_tokens=True)
        model.train(); model.config.use_cache = False
        return jiwer.wer([lang.norm(r["text"]) or "-" for r in rows], [lang.norm(h) or "-" for h in hyps])

    dev_sets = {}
    for p in sorted(data.glob("eval_*_dev.jsonl")):
        rows = read_jsonl(p); random.Random(1).shuffle(rows)
        dev_sets[p.stem[5:-4]] = rows[: a.dev_clips]
    test_sets = {p.stem[5:-5]: read_jsonl(p) for p in sorted(data.glob("eval_*_test.jsonl"))}
    if not dev_sets:       # no dev split: hold out a few training clips so checkpoints can still be compared
        random.Random(1).shuffle(items)
        n = min(a.dev_clips, max(1, len(items) // 20))
        dev_sets["heldout"], items = items[:n], items[n:]
        print(f"no eval_*_dev.jsonl: using {n} held-out training clips as dev", flush=True)

    def dev_score(m):
        scores = {k: eval_set(v, m) for k, v in dev_sets.items()}
        mean = sum(scores.values()) / len(scores)
        return mean, ", ".join(f"{k} dev {v * 100:.1f}%" for k, v in scores.items()) + f", mean {mean * 100:.1f}%"

    model = WhisperForConditionalGeneration.from_pretrained(a.base).to(dev)
    model.generation_config = GenerationConfig.from_pretrained(a.base)
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = model.config.mask_feature_prob = a.mask
    if not a.no_ckpt:                      # checkpointing saves memory at ~25-35% extra compute
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if a.probe:
        # Worst case: full batch, every label padded to the longest transcript. Audio is always padded to 30 s, so
        # this is the peak for the whole run; the second step includes the optimizer state.
        maxlen = min(448, max(len(tok(r["text"]).input_ids) for r in items))
        feats = torch.randn(a.batch, model.config.num_mel_bins, 3000, device=dev)
        labels = torch.randint(0, 50000, (a.batch, maxlen - 1), device=dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                loss = model(input_features=feats, labels=labels).loss
            loss.backward(); opt.step(); opt.zero_grad()
        peak = torch.cuda.max_memory_reserved() / 1024 ** 3 if cuda else float("nan")
        print(f"PROBE peak_reserved_gb={peak:.2f} maxlen={maxlen} batch={a.batch} ckpt={not a.no_ckpt}", flush=True)
        return

    teacher, tmap, tfe = None, None, None
    if a.teacher:
        teacher = WhisperForConditionalGeneration.from_pretrained(
            a.teacher, dtype=torch.bfloat16 if cuda else torch.float32).to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        try:
            ttok = WhisperTokenizer.from_pretrained(a.teacher)
        except OSError:   # fine-tunes sometimes ship without tokenizer files: large-v3 family shares one vocabulary
            ttok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
        tmap = token_map(tok, ttok, model.config.vocab_size).to(dev)
        if bool((tmap == torch.arange(len(tmap), device=dev)).all()) and teacher.config.vocab_size == model.config.vocab_size:
            tmap = None                    # same vocabulary: no remapping needed
        if teacher.config.num_mel_bins != model.config.num_mel_bins:
            tfe = WhisperFeatureExtractor(feature_size=teacher.config.num_mel_bins)
        print(f"teacher: {a.teacher} (vocab remap: {tmap is not None}, own {teacher.config.num_mel_bins}-bin features: "
              f"{tfe is not None})", flush=True)

    def train_loss(feats, tfeats, labels):
        out_ = model(input_features=feats, labels=labels)
        if teacher is None:
            return out_.loss
        mask = labels != -100
        tlabels = labels if tmap is None else torch.where(mask, tmap[labels.clamp(min=0)], labels)
        tin = (tfeats if tfeats is not None else feats).to(teacher.dtype)
        with torch.no_grad():
            tl = teacher(input_features=tin, labels=tlabels).logits
        if tmap is not None:
            tl = tl.index_select(-1, tmap)          # teacher's distribution over the student's vocabulary
        s = torch.log_softmax(out_.logits[mask].float() / KD_T, -1)
        t = torch.log_softmax(tl[mask].float() / KD_T, -1)
        kl = torch.nn.functional.kl_div(s, t, log_target=True, reduction="batchmean") * KD_T ** 2
        return KD_CE * out_.loss + KD_KL * kl

    workers = a.workers if a.workers >= 0 else (6 if cuda else 0)
    loader = DataLoader(ClipDataset(items, fe, tok, tfe), batch_size=a.batch, shuffle=True, num_workers=workers,
                        collate_fn=collate, persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = int(a.epochs * len(loader)) // a.accum
    if a.max_steps:
        total = min(total, a.max_steps)
    total = max(total, 1)
    warmup = min(a.warmup, max(1, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.02, (total - s) / total))
    best, desc = dev_score(model)
    stock, start_desc = best, desc
    print(f"start: {desc} (stock) | {total} steps", flush=True)
    model.save_pretrained(out); processor.save_pretrained(out)     # --out always holds the best so far, stock included
    history = []
    step, micro, t0, running = 0, 0, time.time(), 0.0
    model.train()
    while step < total:
        for feats, tfeats, labels in loader:
            if (labels[:, 0] == model.config.decoder_start_token_id).all():
                labels = labels[:, 1:]      # the model prepends it itself
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                loss = train_loss(feats.to(dev), None if tfeats is None else tfeats.to(dev), labels.to(dev))
            (loss / a.accum).backward()
            micro += 1; running += loss.item() / a.accum
            if micro % a.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); sched.step(); optimizer.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{total} loss {running / 100:.3f} ({(time.time() - t0) / 60:.0f} min)", flush=True)
                running = 0.0
            if step % a.eval_every == 0 or step == total:
                wer, desc = dev_score(model)
                history.append({"step": step, "dev_wer": round(wer * 100, 2)})
                mark = ""
                if wer < best:
                    best = wer; model.save_pretrained(out); processor.save_pretrained(out); mark = "  saved"
                print(f"== step {step}: {desc} (best {best * 100:.1f}%){mark}", flush=True)
            if step >= total:
                break

    if best >= stock:
        print("warning: no checkpoint beat the stock model on dev, so --out is the stock model. More data, more epochs "
              "or a different --lr may help.", flush=True)
    best_model = WhisperForConditionalGeneration.from_pretrained(out).to(dev)
    best_model.generation_config = GenerationConfig.from_pretrained(a.base)
    tests = {k: round(eval_set(v, best_model) * 100, 2) for k, v in test_sets.items()}
    print(f"FINAL {out}: " + "  ".join(f"{k} test {v:.1f}%" for k, v in tests.items()), flush=True)
    info = {"language": lang.code, "base": a.base, "epochs": a.epochs, "steps": total, "batch": a.batch, "accum": a.accum,
            "lr": a.lr, "warmup": warmup, "mask": a.mask, "teacher": a.teacher or None,
            "kd": {"T": KD_T, "ce": KD_CE, "kl": KD_KL} if a.teacher else None,
            "clips": dict(Counter(r["src"] for r in items)), "hours": round(hours, 1),
            "dev_sets": {k: len(v) for k, v in dev_sets.items()}, "start": start_desc, "stock_dev_wer": round(stock * 100, 2), "best_dev_wer": round(best * 100, 2),
            "history": history, "test_wer_pytorch": tests, "minutes": round((time.time() - t0) / 60, 1), "device": dev}
    (out / "train_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
