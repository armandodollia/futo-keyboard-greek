"""LoRA fine-tune of Qwen3-1.7B for dictation cleanup (light / rambler), one model for all languages in --work.

Data: <work>/pairs_*.jsonl ({lang, mode, input, output}). Validation: <work>/val_pairs.json is created on the first
run (2%, min 200 pairs) and REUSED afterwards, and never trained on, so later runs are scored on the same held-out
pairs as earlier ones.
Prompt: the app's one short system line with mode + language (cheap to prefill on a phone), the transcript as the user
turn, the cleaned text as the assistant turn; loss only on the assistant tokens. Qwen3 thinking is disabled.
Outputs <out> (merged model, for GGUF export) and <out>-adapter (the LoRA alone, for convert_lora_to_gguf.py).
Usage: python train_cleanup.py --work work --out work/out/cleanup [--epochs 2] [--langs el --rank 16]
       continue training (e.g. after adding correction pairs): --base work/out/cleanup --epochs 1 --old-frac 0.3
       --dry-run: tokenize a few pairs and check the prompt against the app's format, without loading the model
"""
import argparse
import glob
import json
import os
import random
import time

from common import chat_prompt, raw_prompt


def build(tok, r, maxlen):
    p_ids = tok(chat_prompt(tok, r["mode"], r["lang"], r["input"]), add_special_tokens=False).input_ids
    a_ids = tok(r["output"] + tok.eos_token, add_special_tokens=False).input_ids
    return (p_ids + a_ids)[:maxlen], ([-100] * len(p_ids) + a_ids)[:maxlen]


def load_rows(work, langs, old_frac, save_val=True):
    val_path = os.path.join(work, "val_pairs.json")
    val = json.load(open(val_path, encoding="utf-8")) if os.path.exists(val_path) else None
    seen = {(r["mode"], r["input"]) for r in val} if val else set()
    pick = random.Random(5)
    rows = []
    for f in sorted(glob.glob(os.path.join(work, "pairs_*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            if "corrections" not in f and pick.random() > old_frac:
                continue   # continue-training: keep only a share of the older, non-correction pairs
            r = json.loads(l)
            k = (r["mode"], r["input"])
            if langs and r["lang"] not in langs:
                continue
            if r["input"].strip() and r["output"].strip() and k not in seen:
                seen.add(k); rows.append(r)
    random.Random(0).shuffle(rows)
    if val is None:
        n_val = max(200, len(rows) // 50)
        val, rows = rows[:n_val], rows[n_val:]
        if save_val:
            json.dump(val, open(val_path, "w", encoding="utf-8"), ensure_ascii=False)
    return rows, val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="work/out/cleanup-qwen3-1.7b")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B", help="HF id or a previous merged output (continue training)")
    ap.add_argument("--langs", "--lang", default="", help="train on these languages only, e.g. 'el' or 'el,sq'")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--old-frac", type=float, default=1.0, help="share of non-correction pairs to keep")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=384)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    langs = set(filter(None, a.langs.split(",")))
    train, val = load_rows(a.work, langs, a.old_frac, save_val=not a.dry_run)
    by = {}
    for r in train:
        by[(r["lang"], r["mode"])] = by.get((r["lang"], r["mode"]), 0) + 1
    print(f"train {len(train)} pairs {by}; val {len(val)}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    if a.dry_run:
        for r in train[:3]:
            p = chat_prompt(tok, r["mode"], r["lang"], r["input"])
            assert p == raw_prompt(r["mode"], r["lang"], r["input"]), "chat template differs from the app's prompt"
            ids, labels = build(tok, r, a.maxlen)
            print(f"[{r['lang']}/{r['mode']}] {len(ids)} tokens, {sum(x != -100 for x in labels)} trained\n{p}{r['output']}")
        print("dry run ok: prompts match the app's format")
        return

    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("warning: no CUDA GPU, training on CPU will take days", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    data = [build(tok, r, a.maxlen) for r in train]

    def collate(batch):
        n = max(len(i) for i, _ in batch)
        ids = torch.full((len(batch), n), tok.pad_token_id); lab = torch.full((len(batch), n), -100)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for k, (i, l) in enumerate(batch):
            ids[k, :len(i)] = torch.tensor(i); lab[k, :len(l)] = torch.tensor(l); att[k, :len(i)] = 1
        return ids, lab, att

    loader = DataLoader(data, batch_size=a.batch, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    total = a.epochs * len(loader) // a.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / total))
    print(f"{total} optimizer steps", flush=True)
    model.train(); step = 0; t0 = time.time(); run = 0.0
    for epoch in range(a.epochs):
        for i, (ids, lab, att) in enumerate(loader):
            with torch.autocast(dev, dtype=torch.bfloat16):
                loss = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev)).loss / a.accum
            loss.backward(); run += loss.item()
            if (i + 1) % a.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 25 == 0:
                print(f"epoch {epoch + 1} step {step}/{total} loss {run / 25:.3f} ({(time.time() - t0) / 60:.0f} min)", flush=True)
                run = 0.0
    model.save_pretrained(a.out + "-adapter")   # the LoRA alone (llama.cpp --lora / convert_lora_to_gguf.py)
    merged = model.merge_and_unload()
    merged.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
