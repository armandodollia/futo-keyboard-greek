"""Export a trained cleanup model to GGUF with llama.cpp's tools.

Merged model (what the app imports): convert_hf_to_gguf.py -> f16 GGUF -> llama-quantize Q4_0 (~1 GB for 1.7B).
Q4_0 because llama.cpp repacks it at load time into its fastest ARM (dotprod/i8mm) kernels, which is what phones run.
LoRA adapter (--adapter): convert_lora_to_gguf.py -> a small f16 GGUF for `llama-server -m base.gguf --lora x.gguf`.
The app runs a merged model (import or download) or the stock base + a per-language adapter (download index, kind
cleanup-adapter); adapters must be trained on the stock base (Qwen/Qwen3-1.7B).

Needs a llama.cpp checkout with built tools (llama-quantize) and its Python deps:
  pip install -r <llama.cpp>/requirements/requirements-convert_hf_to_gguf.txt
Usage: python export_gguf.py --llama-cpp ../llama.cpp --model work/out/cleanup-qwen3-1.7b --out work/cleanup-q4_0.gguf
       python export_gguf.py --llama-cpp ../llama.cpp --adapter work/out/cleanup-el-adapter --base Qwen/Qwen3-1.7B \
         --out work/cleanup-el-lora.gguf
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys


def tool(root, name):
    exe = name + (".exe" if os.name == "nt" else "")
    for sub in ("", "bin", "build/bin", "build/bin/Release", "build/bin/Debug"):
        p = pathlib.Path(root) / sub / exe
        if p.exists():
            return str(p)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} not found under {root}: build llama.cpp (cmake -B build && cmake --build build "
                         "--config Release) or put it on PATH")
    return found


def run(cmd):
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", required=True, help="llama.cpp checkout")
    ap.add_argument("--model", help="merged HF model folder (train_cleanup.py --out)")
    ap.add_argument("--adapter", help="LoRA adapter folder (<out>-adapter)")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B", help="adapter's base: local folder or HF id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="Q4_0", help="llama-quantize type for --model (Q4_0 for the app)")
    ap.add_argument("--keep-f16", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(a.llama_cpp)
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    if a.adapter:
        base_dir = a.base
        if not pathlib.Path(base_dir).is_dir():
            # Fetch the base's config/tokenizer (no weights) and pass a folder: with --base-model-id,
            # convert_lora_to_gguf.py names the adapter after the base ("general.name = Qwen/Qwen3-1.7B") instead of
            # after the adapter folder like the published adapters ("Cleanup El Adapter").
            try:
                from huggingface_hub import snapshot_download
                base_dir = snapshot_download(a.base, allow_patterns=["*.json", "*.txt"])
            except Exception as e:  # noqa: BLE001
                print(f"could not fetch {a.base} ({e}); falling back to --base-model-id", flush=True)
        base = ["--base", base_dir] if pathlib.Path(base_dir).is_dir() else ["--base-model-id", a.base]
        run([sys.executable, str(root / "convert_lora_to_gguf.py"), a.adapter, *base, "--outtype", "f16",
             "--outfile", str(out)])
    elif a.model:
        f16 = out.with_name(out.stem + "-f16.gguf")
        run([sys.executable, str(root / "convert_hf_to_gguf.py"), a.model, "--outtype", "f16", "--outfile", str(f16)])
        run([tool(root, "llama-quantize"), str(f16), str(out), a.quant])
        if not a.keep_f16:
            f16.unlink()
    else:
        raise SystemExit("give --model (merged) or --adapter")
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
