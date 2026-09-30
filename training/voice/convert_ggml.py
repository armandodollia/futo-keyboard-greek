"""Convert a Hugging Face Whisper folder to a whisper.cpp ggml .bin (the format FUTO Keyboard imports), then quantize.

The writer follows whisper.cpp's models/convert-h5-to-ggml.py (MIT) field for field, so it needs neither a whisper.cpp
nor an openai/whisper checkout: the mel filterbank comes from transformers' feature extractor (the same Slaney
filters as openai/whisper's mel_filters.npz), and the vocabulary from the model folder or the base model on the Hub.
Quantization needs whisper.cpp's `whisper-quantize` binary. q8_0 is what shipped for Greek and Albanian: half the size
of f16 (small: 252 MB vs 465 MB) with no measurable WER change; q5_1 is smaller still but untested here.

Usage: python convert_ggml.py <model dir> <out.bin> [--quantize q8_0] [--whisper-cpp DIR]
"""
import argparse
import json
import pathlib
import struct
import subprocess

import numpy as np

from common import find_tool

CONV = {
    "self_attn.k_proj": "attn.key", "self_attn.q_proj": "attn.query", "self_attn.v_proj": "attn.value",
    "self_attn.out_proj": "attn.out", "self_attn_layer_norm": "attn_ln",
    "encoder_attn.q_proj": "cross_attn.query", "encoder_attn.v_proj": "cross_attn.value",
    "encoder_attn.out_proj": "cross_attn.out", "encoder_attn_layer_norm": "cross_attn_ln",
    "fc1": "mlp.0", "fc2": "mlp.2", "final_layer_norm": "mlp_ln",
    "encoder.layer_norm.bias": "encoder.ln_post.bias", "encoder.layer_norm.weight": "encoder.ln_post.weight",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.layer_norm.bias": "decoder.ln.bias", "decoder.layer_norm.weight": "decoder.ln.weight",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
}


def bytes_to_unicode():
    """GPT-2 byte <-> printable-unicode table: vocab.json stores tokens in this form, ggml stores raw bytes."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


def ggml_name(name):
    parts = name.split(".")[1:]              # drop the leading "model."
    if parts[1] == "layers":
        parts[1] = "blocks"
        key = ".".join(parts[3:-1])
        mapped = ("attn.key" if parts[0] == "encoder" else "cross_attn.key") if key == "encoder_attn.k_proj" else CONV[key]
        return ".".join(parts[:3] + [mapped] + parts[-1:])
    name = ".".join(parts)
    return CONV.get(name, name)


def load_vocab(model_dir: pathlib.Path, vocab_from: str):
    if (model_dir / "vocab.json").exists():
        return json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download     # newer transformers save only tokenizer.json
    return json.loads(pathlib.Path(hf_hub_download(vocab_from, "vocab.json")).read_text(encoding="utf-8"))


def convert(model_dir, out_path, vocab_from="openai/whisper-small"):
    import torch
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration
    model_dir, out_path = pathlib.Path(model_dir), pathlib.Path(out_path)
    hp = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    max_length = hp.get("max_length") or hp.get("max_target_positions", 448)
    n_mels = hp["num_mel_bins"]
    filters = WhisperFeatureExtractor(feature_size=n_mels).mel_filters.T.astype(np.float32)   # (n_mels, 201)
    tokens = sorted(load_vocab(model_dir, vocab_from).items(), key=lambda kv: kv[1])
    model = WhisperForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32)
    dec = {v: k for k, v in bytes_to_unicode().items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("i", 0x67676D6C))       # "ggml"
        for v in (hp["vocab_size"], hp["max_source_positions"], hp["d_model"], hp["encoder_attention_heads"],
                  hp["encoder_layers"], int(max_length), hp["d_model"], hp["decoder_attention_heads"],
                  hp["decoder_layers"], n_mels, 1):  # last = f16
            f.write(struct.pack("i", v))
        f.write(struct.pack("ii", *filters.shape))
        f.write(filters.tobytes())
        f.write(struct.pack("i", len(tokens)))
        for tok, _ in tokens:
            b = bytes(dec[c] for c in tok)
            f.write(struct.pack("i", len(b))); f.write(b)
        for name, t in model.state_dict().items():
            if name == "proj_out.weight":            # tied to the token embedding; whisper.cpp reuses that
                continue
            gname = ggml_name(name)
            data = t.squeeze().numpy().astype(np.float16)
            if gname in ("encoder.conv1.bias", "encoder.conv2.bias"):
                data = data.reshape(data.shape[0], 1)
            # 1-D tensors, conv biases and positional embeddings stay f32 (as whisper.cpp expects)
            f32 = data.ndim < 2 or gname in ("encoder.conv1.bias", "encoder.conv2.bias",
                                             "encoder.positional_embedding", "decoder.positional_embedding")
            data = data.astype(np.float32) if f32 else data
            nb = gname.encode("utf-8")
            f.write(struct.pack("iii", data.ndim, len(nb), 0 if f32 else 1))
            for d in reversed(data.shape):
                f.write(struct.pack("i", d))
            f.write(nb)
            f.write(data.tobytes())
    print(f"wrote {out_path} ({out_path.stat().st_size / 2 ** 20:.0f} MB, f16)", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Hugging Face Whisper folder (normally the ACFT output)")
    ap.add_argument("out", help="output .bin (quantized if --quantize, else f16)")
    ap.add_argument("--quantize", default="q8_0", help="whisper.cpp quant type, or 'none' for f16 only")
    ap.add_argument("--whisper-cpp", default="", help="whisper.cpp checkout or bin folder (for whisper-quantize)")
    ap.add_argument("--vocab-from", default="openai/whisper-small",
                    help="where to get vocab.json if the folder lacks it (all multilingual Whisper sizes share it)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if a.quantize == "none":
        convert(a.model, out, a.vocab_from)
        return
    quant = find_tool("whisper-quantize", a.whisper_cpp)      # fail before the slow part if it's missing
    f16 = out.with_name(out.stem + "-f16.bin")
    convert(a.model, f16, a.vocab_from)
    subprocess.run([quant, str(f16), str(out), a.quantize], check=True, capture_output=True)
    print(f"wrote {out} ({out.stat().st_size / 2 ** 20:.0f} MB, {a.quantize})", flush=True)


if __name__ == "__main__":
    main()
