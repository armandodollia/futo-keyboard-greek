"""Clean the raw text, train the SentencePiece tokenizer, and tokenize everything.

Input : <raw>/<source>.txt for every source in languages/<code>.yaml (local sources: their own `path`)
Output: <tok>/
          tokenizer.model / .vocab      SentencePiece unigram (16k), whitespace as suffix (FUTO "inverted_space"),
                                        <XBU> <XBC> <XEC> user symbols, unk=0 bos=1 eos=2 pad=3
          <source>.clean.txt            cleaned, deduplicated lines
          <source>.train.bin / .idx     uint16 token ids of every kept line, and line start offsets (uint64)
          <source>.val.txt              held-out lines (0.2%, max 5k) for evaluation
          stats.json

Usage: python prepare.py --lang xx --raw work/xx/raw --tok work/xx/tok [--max-lines N] [--skip-clean] [--skip-sp]
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import pathlib
import random
import re
import unicodedata

import numpy as np
import sentencepiece as spm

import langconfig

LETTER = re.compile(r"[^\W\d_]")
URL = re.compile(r"(https?://|www\.)\S+|\S+@\S+\.\S+")
TAGS = re.compile(r"<[^>]{1,20}>|\{[^}]{1,40}\}|[♪♫#]")
G = {}  # per-process state (set by setup; the pool re-runs it in every worker, also under Windows' spawn)


def setup(lang, raw, tok, max_lines):
    cfg = langconfig.load(lang)
    end = "".join(re.escape(c) for c in cfg["sentence_end"])
    G.update(cfg=cfg, raw=pathlib.Path(raw), tok=pathlib.Path(tok), max_lines=max_lines,
             script=re.compile(f"[{cfg['letters']}]"),
             sent=re.compile(f"(?<=[{end}])\\s+(?=[\"«(“{cfg.get('sentence_start', cfg['letters'])}])"))


def clean(line, src):
    cfg, conf = G["cfg"], G["cfg"]["sources"][src]
    line = unicodedata.normalize("NFC", line.strip())
    if conf["from"] == "opensubtitles":
        line = TAGS.sub("", line).lstrip("-– ").strip()
    line = URL.sub("", line)
    line = re.sub(r"\s+", " ", line).strip()
    if len(line) < conf.get("min_chars", 15) or len(line) > 2000:
        return []
    letters = LETTER.findall(line)
    if not letters or len(G["script"].findall(line)) < cfg["min_script_share"] * len(letters):
        return []
    out, buf = [], ""
    for s in G["sent"].split(line):  # long paragraphs -> sentence groups of <= ~300 chars
        if buf and len(buf) + len(s) > 300:
            out.append(buf); buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        out.append(buf)
    return out


def raw_lines(src):
    conf = G["cfg"]["sources"][src]
    if conf["from"] == "local":
        for path in sorted(glob.glob(str(pathlib.Path(G["cfg"]["_dir"]) / conf["path"]))):
            yield from open(path, encoding="utf-8", errors="replace")
    else:
        yield from open(G["raw"] / f"{src}.txt", encoding="utf-8", errors="replace")


def clean_source(src):
    out = G["tok"] / f"{src}.clean.txt"
    seen, kept, n = set(), 0, 0
    keep, rng = G["cfg"]["sources"][src].get("keep", 1.0), random.Random(src)
    with open(out, "w", encoding="utf-8") as f:
        for raw in raw_lines(src):
            if G["max_lines"] and n >= G["max_lines"]:
                break
            n += 1
            if keep < 1.0 and rng.random() > keep:
                continue
            for s in clean(raw, src):
                h = hashlib.blake2b(s.lower().encode(), digest_size=8).digest()
                if h in seen:
                    continue
                seen.add(h); f.write(s + "\n"); kept += 1
    print(f"{src}: {n} raw lines -> {kept} clean lines", flush=True)
    return src, kept


def train_tokenizer(counts):
    # sample lines per the `sp` weights, so informal text is over-represented and slang gets its own pieces
    cfg, tok = G["cfg"], G["tok"]
    sp_lines = int(cfg["tokenizer"]["sample_lines"])
    sample = tok / "sp_sample.txt"
    rng = random.Random(0)
    with open(sample, "w", encoding="utf-8") as f:
        for src, c in counts.items():
            want = int(sp_lines * cfg["sources"][src].get("sp", 1 / len(counts)))
            p = min(1.0, want / max(1, c))
            reps = min(20, max(1, int(want / max(1, c))))  # small sources repeated, but capped: hundreds of identical lines make SentencePiece crawl
            for line in open(tok / f"{src}.clean.txt", encoding="utf-8"):
                for _ in range(reps if p >= 1 else 1):
                    if p >= 1 or rng.random() < p:
                        f.write(line)
    spm.SentencePieceTrainer.Train(
        input=str(sample), model_prefix=str(tok / "tokenizer"), vocab_size=int(cfg["tokenizer"]["vocab"]),
        model_type="unigram", character_coverage=0.99995, treat_whitespace_as_suffix=True, byte_fallback=True,
        split_by_unicode_script=True, split_by_whitespace=True, split_by_number=True, max_sentencepiece_length=32,
        user_defined_symbols=["<XBU>", "<XBC>", "<XEC>"], unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        input_sentence_size=sp_lines, shuffle_input_sentence=True, num_threads=os.cpu_count(),
        normalization_rule_name="identity", minloglevel=1)
    sample.unlink()


def _encode(chunk):
    sp = spm.SentencePieceProcessor(model_file=str(G["tok"] / "tokenizer.model"))
    return sp.encode([l.rstrip("\n") + " " for l in chunk])


def tokenize_source(src, pool):
    tok = G["tok"]
    lines = open(tok / f"{src}.clean.txt", encoding="utf-8").readlines()
    random.Random(1).shuffle(lines)
    n_val = min(5000, len(lines) // 500) if G["cfg"]["sources"][src].get("val", True) else 0
    (tok / f"{src}.val.txt").write_text("".join(lines[:n_val]), encoding="utf-8")
    train = lines[n_val:]
    chunks = [train[i:i + 20000] for i in range(0, len(train), 20000)]
    ids, idx, total = open(tok / f"{src}.train.bin", "wb"), [0], 0
    for enc in pool.imap(_encode, chunks):
        for seq in enc:
            ids.write(np.asarray(seq, dtype=np.uint16).tobytes()); total += len(seq); idx.append(total)
    ids.close()
    np.asarray(idx, dtype=np.uint64).tofile(tok / f"{src}.train.idx")
    print(f"{src}: {len(train)} train lines, {total / 1e6:.1f}M tokens, {n_val} val lines", flush=True)
    return len(train), total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument("--max-lines", type=int, default=0, help="read at most N raw lines per source (quick tests)")
    p.add_argument("--skip-clean", action="store_true", help="reuse *.clean.txt from an earlier run")
    p.add_argument("--skip-sp", action="store_true", help="reuse an existing tokenizer.model")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count()), help="tokenizer processes (each holds a chunk in RAM)")
    args = p.parse_args()
    pathlib.Path(args.tok).mkdir(parents=True, exist_ok=True)
    init = (args.lang, args.raw, args.tok, args.max_lines)
    setup(*init)
    cfg, tok = G["cfg"], G["tok"]

    def available(s):
        if cfg["sources"][s]["from"] == "local":
            return bool(glob.glob(str(pathlib.Path(cfg["_dir"]) / cfg["sources"][s]["path"])))
        return (G["raw"] / f"{s}.txt").exists() or (args.skip_clean and (tok / f"{s}.clean.txt").exists())
    srcs = [s for s in cfg["sources"] if available(s)]
    for s in cfg["sources"]:
        if s not in srcs:
            print(f"{s}: no data, skipped", flush=True)
    assert srcs, "no source has data: run the download stage first"
    with mp.Pool(max(1, min(args.workers, os.cpu_count())), initializer=setup, initargs=init) as pool:
        if args.skip_clean:
            counts = {s: sum(1 for _ in open(tok / f"{s}.clean.txt", encoding="utf-8")) for s in srcs}
        else:
            counts = dict(pool.map(clean_source, srcs))
        srcs = [s for s in srcs if counts[s]]
        counts = {s: counts[s] for s in srcs}
        if not (args.skip_sp and (tok / "tokenizer.model").exists()):
            train_tokenizer(counts)
        stats = {s: dict(zip(("lines", "tokens"), tokenize_source(s, pool))) for s in srcs}
    sp = spm.SentencePieceProcessor(model_file=str(tok / "tokenizer.model"))
    alphabet = cfg.get("alphabet", "")
    # every lowercase letter needs its own piece: the keyboard maps each key to the letter's single-character token
    # (capitals fall back to the lowercase token)
    stats["single_letter_pieces_missing"] = [c for c in alphabet if sp.piece_to_id(c) == sp.unk_id()]
    stats["examples"] = {w: sp.encode_as_pieces(w + " ") for w in cfg["tokenizer"].get("examples", [])}
    (tok / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    try:
        langconfig.Language(cfg).bind(sp)
    except AssertionError as e:
        raise SystemExit(f"ERROR: {e}. Add text with those letters (or check layout.rows) and prepare again.")


if __name__ == "__main__":
    main()
