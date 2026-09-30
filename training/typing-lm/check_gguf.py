"""Run GGUF files in llama.cpp (llama-cpp-python) and compare their logits with the PyTorch model.

This is the check that catches export bugs FUTO would silently live with (e.g. a missing Q/K RoPE permutation: the
model still loads and suggests words, just worse). Expect argmax agreement ~100% for F16 and >= ~95% for Q6_K.

Usage: python check_gguf.py <model dir> <file.gguf> [<file.gguf> ...]      (pip install llama-cpp-python)
"""
import sys

import numpy as np
import torch
from transformers import LlamaForCausalLM

import llama_cpp


def main(model_dir, files):
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    ids = torch.randint(4, model.config.vocab_size, (1, 64), generator=torch.Generator().manual_seed(0))
    ids[0, 0] = 1  # BOS
    with torch.no_grad():
        ref = model(ids).logits[0].numpy()
    worst = 1.0
    for f in files:
        m = llama_cpp.Llama(model_path=f, n_ctx=128, logits_all=True, verbose=False)
        m.eval(ids[0].tolist())
        got = np.array(m.scores[:ids.shape[1]])
        agree = (got.argmax(-1) == ref.argmax(-1)).mean()
        worst = min(worst, agree)
        print(f"{f}: argmax agree {agree:.0%}  max|dlogit| {np.abs(got - ref).max():.3f}", flush=True)
    if worst < 0.8:
        sys.exit("llama.cpp disagrees with PyTorch: the export is broken")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
