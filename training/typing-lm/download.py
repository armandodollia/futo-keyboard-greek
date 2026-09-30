"""Download raw text for a language, as listed under `sources:` in languages/<code>.yaml.

Output: <out>/<source>.txt, one paragraph / sentence / subtitle line per line, UTF-8. Each source is independent: a
failure is logged and the others continue. Local sources (`from: local`) are read directly by prepare.py.

  fineweb2       HuggingFaceFW/fineweb-2 data/<dataset>/train/*.parquet (web; ODC-By), capped at max_mb
  wikipedia      wikimedia/wikipedia <dump>.<wiki> (CC BY-SA)
  opensubtitles  OPUS OpenSubtitles v2024 (fallback v2018) monolingual <opus>.txt.gz. Personal use only: the
                 subtitles are copyrighted, so think twice before publishing a model trained on them.
  huggingface    any HF dataset, via the parquet files the Hub makes for it (columns joined with a space)
  url            a plain-text or .gz file

Usage: python download.py --lang xx --out work/xx/raw [source ...]
"""
import argparse
import gzip
import io
import json
import os
import pathlib
import shutil
import traceback
import urllib.request

import langconfig


def log(*a):
    print(*a, flush=True)


def write_paras(f, text, min_len=1):
    n = 0
    for para in str(text).split("\n"):
        para = para.strip()
        if len(para) >= min_len:
            f.write(para + "\n"); n += len(para.encode())
    return n


def hf_parquet_files(repo, prefix):
    from huggingface_hub import HfApi
    return sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                  if f.startswith(prefix) and f.endswith(".parquet"))


def parquet_text(repo, files, columns, out, cap):
    """Stream the text columns of HF parquet shards into out; each shard is deleted after use (keeps disk low)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for name in files:
            p = hf_hub_download(repo, name, repo_type="dataset")
            for batch in pq.ParquetFile(p).iter_batches(columns=columns, batch_size=2000):
                for row in zip(*(batch.column(c).to_pylist() for c in columns)):
                    written += write_paras(f, " ".join(str(x) for x in row if x), 21 if repo.endswith("fineweb-2") else 1)
                if written >= cap:
                    break
            for f in {os.path.realpath(p), p}:  # the snapshot path is a symlink to the blob on macOS/Linux: remove both
                try:
                    os.remove(f)
                except OSError:
                    pass
            log(f"{out.stem}: {name} -> {written / 1e6:.0f} MB")
            if written >= cap:
                break


def fineweb2(src, cfg, out):
    files = hf_parquet_files("HuggingFaceFW/fineweb-2", f"data/{src['dataset']}/train/")
    assert files, f"no FineWeb-2 shards for {src['dataset']}"
    log(f"{out.stem}: {len(files)} shards")
    parquet_text("HuggingFaceFW/fineweb-2", files, ["text"], out, src.get("max_mb", 1e9) * 1e6)


def wikipedia(src, cfg, out):
    config = f"{src.get('dump', '20231101')}.{src.get('wiki', cfg['code'])}"
    files = hf_parquet_files("wikimedia/wikipedia", f"{config}/")
    assert files, f"no wikimedia/wikipedia config {config}"
    parquet_text("wikimedia/wikipedia", files, ["text"], out, src.get("max_mb", 1e9) * 1e6)


def opensubtitles(src, cfg, out):
    code = src.get("opus", cfg["code"])
    for ver in ("v2024", "v2018"):
        url = f"https://object.pouta.csc.fi/OPUS-OpenSubtitles/{ver}/mono/{code}.txt.gz"
        try:
            log(f"{out.stem}: {url}")
            fetch_text(url, out)
            return
        except Exception as e:  # noqa: BLE001
            log(f"{out.stem}: {url} failed: {e}")
    raise RuntimeError(f"no OpenSubtitles monolingual file for '{code}'")


def fetch_text(url, out):
    tmp = out.with_suffix(".download")
    urllib.request.urlretrieve(url, tmp)
    with (gzip.open(tmp, "rb") if url.endswith(".gz") else open(tmp, "rb")) as s, open(out, "wb") as d:
        shutil.copyfileobj(s, d)
    tmp.unlink()


def huggingface(src, cfg, out):
    """Via the Hub's parquet API: works for script-based datasets too (the Hub converts them)."""
    info = json.loads(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{src['repo']}/parquet").read())
    configs = [src["config"]] if src.get("config") else list(info)
    split = src.get("split", "train")  # "all": every split (small labelled sets are often split train/test)
    urls = [u for c in configs for sp, us in info[c].items() if split in ("all", sp) for u in us]
    assert urls, f"{src['repo']}: nothing for config {configs} split {src.get('split', 'train')}"
    import pyarrow.parquet as pq
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for u in urls:
            t = pq.read_table(io.BytesIO(urllib.request.urlopen(u).read()))
            cols = [c for c in src.get("columns", ["text"]) if c in t.column_names] or [t.column_names[1]]
            for row in zip(*(t.column(c).to_pylist() for c in cols)):
                line = " ".join(str(x) for x in row if x).replace("\n", " ").strip()
                if line:
                    f.write(line + "\n"); n += 1
    log(f"{out.stem}: {n} lines")


JOBS = {"fineweb2": fineweb2, "wikipedia": wikipedia, "opensubtitles": opensubtitles, "huggingface": huggingface,
        "url": lambda src, cfg, out: fetch_text(src["url"], out)}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, help="language code or config path")
    p.add_argument("--out", required=True, help="raw text directory")
    p.add_argument("--force", action="store_true", help="download again even if <source>.txt exists")
    p.add_argument("only", nargs="*", help="only these sources")
    args = p.parse_args()
    cfg = langconfig.load(args.lang)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, src in cfg["sources"].items():
        if (args.only and name not in args.only) or src["from"] == "local":
            continue
        dst = out / f"{name}.txt"
        if dst.exists() and not args.force:
            log(f"{name}: exists ({dst.stat().st_size / 1e6:.0f} MB), skipped")
            continue
        try:
            JOBS[src["from"]](src, cfg, dst)
            log(f"{name}: {dst.stat().st_size / 1e6:.0f} MB")
        except Exception:  # noqa: BLE001
            log(f"{name}: FAILED"); traceback.print_exc(); failed.append(name)
            dst.unlink(missing_ok=True)
    log("downloads finished" + (f"; failed: {' '.join(failed)}" if failed else ""))
