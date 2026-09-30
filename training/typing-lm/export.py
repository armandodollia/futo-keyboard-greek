"""Export a trained model to a FUTO Keyboard GGUF (F16). Quantize afterwards with llama-quantize (pipeline.py does).

Metadata FUTO reads (layout as in stamchry/greek-keyboard-lm's exporter, Apache-2.0):
  keyboardlm.languages            the language code: the keyboard offers the model for that language
  keyboardlm.features             base_v1 inverted_space xbu_char_autocorrect_v1 char_embed_mixing_v1 lora_finetunable_v1
                                  (inverted_space: pieces end in "▁"; xbu_char_autocorrect_v1: <XBU> keys <XBC> input;
                                  char_embed_mixing_v1: one mixed embedding per keystroke)
  keyboardlm.ext_tokenizer_type   sentencepiece, with the raw tokenizer.model in keyboardlm.ext_tokenizer_data

Two things llama.cpp needs that a plain dump of the HF weights gets wrong:
  - Q/K permutation. HF Llama applies RoPE to dimension pairs (i, i + d/2); llama.cpp's LLaMA graph (also FUTO's copy,
    rope mode 0) rotates adjacent pairs (2i, 2i + 1). llama.cpp's own converter reorders the rows of attn_q/attn_k
    for this; without it the model runs, but with every position encoding scrambled.
  - output.weight: the embeddings are tied, and FUTO's loader wants the output matrix as its own tensor.

--check (default) reads the file back, undoes the permutation and compares logits with the PyTorch model.

Usage: python export.py <model dir> <out.gguf> [--lang xx] [--author NAME] [--license Apache-2.0]
"""
import argparse
import pathlib

import gguf
import numpy as np
import sentencepiece as spm
import torch
from transformers import LlamaForCausalLM

import langconfig

FEATURES = "base_v1 inverted_space xbu_char_autocorrect_v1 char_embed_mixing_v1 lora_finetunable_v1"
NAMES = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
         "lm_head.weight": "output.weight"}
LAYER = {"self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
         "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
         "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
         "mlp.down_proj.weight": "ffn_down.weight", "input_layernorm.weight": "attn_norm.weight",
         "post_attention_layernorm.weight": "ffn_norm.weight"}


def gguf_name(name):
    if name in NAMES:
        return NAMES[name]
    _, _, i, rest = name.split(".", 3)
    return f"blk.{i}.{LAYER[rest]}"


def permute(w, n_head):
    """HF (rotate-half) -> GGML (interleaved) RoPE row order; same as llama.cpp convert_hf_to_gguf.py."""
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def unpermute(w, n_head):
    return w.reshape(n_head, w.shape[0] // n_head // 2, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def export(model_dir, out, cfg, author, license_, name=None):
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    c = model.config
    tok_path = pathlib.Path(model_dir) / "tokenizer.model"
    sp = spm.SentencePieceProcessor(model_file=str(tok_path))
    assert sp.get_piece_size() == c.vocab_size, "tokenizer and model vocab differ"

    w = gguf.GGUFWriter(str(out), "llama")
    w.add_name(name or f"{cfg['name']} Typing LM (FUTO, keystroke-mixing)")
    w.add_description(f"{cfg['name']} next-word + autocorrect LM for FUTO Keyboard")
    w.add_author(author)
    w.add_license(license_)
    w.add_block_count(c.num_hidden_layers)
    w.add_context_length(c.max_position_embeddings)
    w.add_embedding_length(c.hidden_size)
    w.add_feed_forward_length(c.intermediate_size)
    w.add_head_count(c.num_attention_heads)
    w.add_head_count_kv(c.num_key_value_heads)
    w.add_layer_norm_rms_eps(c.rms_norm_eps)
    w.add_rope_dimension_count(c.hidden_size // c.num_attention_heads)
    rope = getattr(c, "rope_theta", None) or (getattr(c, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
    w.add_rope_freq_base(float(rope))
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    w.add_string("keyboardlm.languages", cfg["code"])
    w.add_string("keyboardlm.features", FEATURES)
    w.add_string("keyboardlm.ext_tokenizer_type", "sentencepiece")
    # must be bytes (-> UINT8 array): FUTO reads the array's raw data as the serialized SentencePiece model
    w.add_array("keyboardlm.ext_tokenizer_data", tok_path.read_bytes())
    w.add_uint32("keyboardlm.finetuning_count", 0)
    w.add_string("keyboardlm.history", "")

    # llama.cpp's own vocab (FUTO tokenizes with ext_tokenizer_data, but the loader needs a valid SPM vocab)
    types = []
    for i in range(sp.get_piece_size()):
        types.append(gguf.TokenType.UNKNOWN if sp.is_unknown(i) else gguf.TokenType.CONTROL if sp.is_control(i)
                     else gguf.TokenType.BYTE if sp.is_byte(i)
                     else gguf.TokenType.USER_DEFINED if sp.id_to_piece(i) in ("<XBU>", "<XBC>", "<XEC>")
                     else gguf.TokenType.NORMAL)
    w.add_tokenizer_model("llama")
    w.add_token_list([sp.id_to_piece(i) for i in range(sp.get_piece_size())])
    w.add_token_scores([sp.get_score(i) for i in range(sp.get_piece_size())])
    w.add_token_types(types)
    w.add_bos_token_id(1); w.add_eos_token_id(2); w.add_unk_token_id(0); w.add_pad_token_id(3)
    w.add_add_bos_token(True); w.add_add_eos_token(False)

    sd = {k: v for k, v in model.state_dict().items() if "rotary_emb" not in k}
    sd.setdefault("lm_head.weight", sd["model.embed_tokens.weight"])  # tied: write output.weight explicitly
    for k, t in sd.items():
        a = t.numpy()
        if k.endswith("q_proj.weight"):
            a = permute(a, c.num_attention_heads)
        elif k.endswith("k_proj.weight"):
            a = permute(a, c.num_key_value_heads)
        # norms stay F32 (GGML binary ops need it); 2-D weights F16
        w.add_tensor(gguf_name(k), a.astype(np.float32 if a.ndim == 1 else np.float16))
    out.parent.mkdir(parents=True, exist_ok=True)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)  languages={cfg['code']}  features: {FEATURES}", flush=True)
    return model


def check(model, out):
    """Read the GGUF back, undo the permutation, and compare logits with the original model."""
    r = gguf.GGUFReader(str(out))
    field = lambda k: r.fields[k]  # noqa: E731
    lang = bytes(field("keyboardlm.languages").parts[-1]).decode()
    feats = bytes(field("keyboardlm.features").parts[-1]).decode()
    tokf = field("keyboardlm.ext_tokenizer_data")
    tok_ok = tokf.types[-1] == gguf.GGUFValueType.UINT8 and len(tokf.data) == (pathlib.Path(model.name_or_path) / "tokenizer.model").stat().st_size
    c = model.config
    t = {x.name: np.array(x.data) for x in r.tensors}
    sd = {}
    for k in model.state_dict():
        if "rotary_emb" in k:
            continue
        a = t[gguf_name(k)].astype(np.float32).reshape(model.state_dict()[k].shape)
        if k.endswith("q_proj.weight"):
            a = unpermute(a, c.num_attention_heads)
        elif k.endswith("k_proj.weight"):
            a = unpermute(a, c.num_key_value_heads)
        sd[k] = torch.from_numpy(a.copy())
    back = LlamaForCausalLM(c).eval()
    back.load_state_dict(sd, strict=False)
    back.tie_weights()
    ids = torch.randint(4, c.vocab_size, (1, 48), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a, b = model.eval()(ids).logits, back(ids).logits
    diff = (a - b).abs().max().item()
    same = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"check: languages={lang} features='{feats}' tokenizer bytes ok={tok_ok} tensors={len(t)} "
          f"max|dlogit|={diff:.3f} argmax agree={same:.1%}", flush=True)
    assert tok_ok and "output.weight" in t and "char_embed_mixing_v1" in feats and same > 0.9, "GGUF check failed"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("out")
    p.add_argument("--lang", default="", help="language code or config (default: the model dir's language.yaml)")
    p.add_argument("--author", default="FUTO Keyboard Polyglot contributors")
    p.add_argument("--license", default="Apache-2.0", help="of the weights; mind your data's licences (README)")
    p.add_argument("--name", default="")
    p.add_argument("--no-check", action="store_true")
    a = p.parse_args()
    cfg = langconfig.load(a.lang or str(pathlib.Path(a.model_dir) / "language.yaml"))
    m = export(a.model_dir, pathlib.Path(a.out), cfg, a.author, a.license, a.name or None)
    if not a.no_check:
        check(m, pathlib.Path(a.out))
