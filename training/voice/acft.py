"""FUTO ACFT (audio-context fine-tuning), adapted from futo-org/whisper-acft (MIT).

FUTO Keyboard doesn't run the encoder on Whisper's full 30 s window: it runs it only on as many frames as the dictation
needs. A normally fine-tuned model then tends to repeat itself or stop early. ACFT trains the model on shortened
contexts to match (L2 on all decoder hidden states) a frozen copy of itself that sees the full window. Run it after
train.py and before convert_ggml.py; skip it and the .bin works in whisper.cpp but misbehaves in FUTO.

Audio: clips from the language's training manifests (<= 29 s), --max-clips per epoch. Adam, lr 1e-6, 4 epochs, as FUTO.
Greek small: 4 epochs x 3,215 clips took ~24 min on an RTX 4090 (batch 1, as in FUTO's notebook).
"""
import argparse
import pathlib
import random

import torch
from torch import nn

from common import SR, load_audio, load_lang, pick_device, read_jsonl


def encode_partial(model, feats, n_ctx):
    """Run the encoder on the first n_ctx positions only (what whisper.cpp does with -ac / FUTO does per clip)."""
    diff = 2 * n_ctx - feats.shape[2]
    feats = nn.functional.pad(feats, [0, diff, 0, 0, 0, 0]) if diff > 0 else feats[:, :, :diff] if diff < 0 else feats
    if n_ctx == 1500:
        return model.encoder(feats).last_hidden_state
    enc = model.encoder
    x = nn.functional.gelu(enc.conv1(feats))
    x = nn.functional.gelu(enc.conv2(x)).permute(0, 2, 1)
    h = x + enc.embed_positions.weight[:n_ctx]
    h = nn.functional.dropout(h, p=enc.dropout, training=enc.training)
    for layer in enc.layers:
        if enc.training and torch.rand([]) < enc.layerdrop:
            continue
        out = layer(h, attention_mask=None)
        h = out[0] if isinstance(out, (tuple, list)) else out
    return enc.layer_norm(h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--model", required=True, help="fine-tuned model folder (train.py --out)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", help="manifest folder (default work/<code>/data)")
    ap.add_argument("--manifests", nargs="*", help="audio for ACFT (default: <data>/train_*.jsonl)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-clips", type=int, default=4000, help="clips per epoch (random sample)")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperModel, WhisperProcessor
    lang = load_lang(a.lang)
    dev = pick_device(a.device)
    data = pathlib.Path(a.data or f"work/{lang.code}/data")
    rows = [r for f in (a.manifests or sorted(data.glob("train_*.jsonl"))) for r in read_jsonl(f)]
    rows = [r for r in rows if r.get("dur", 0) <= 29]
    if not rows:
        raise SystemExit("no clips for ACFT")

    processor = WhisperProcessor.from_pretrained(a.model)
    processor.tokenizer.set_prefix_tokens(language=lang.whisper_language, task="transcribe")
    model_train = WhisperModel.from_pretrained(a.model).to(dev).train()
    model_base = WhisperModel.from_pretrained(a.model).to(dev).eval()     # frozen full-context reference
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_train.parameters(), lr=a.lr)

    for epoch in range(a.epochs):
        random.Random(epoch).shuffle(rows)
        losses = []
        for i, r in enumerate(rows[: a.max_clips]):
            try:
                wav = load_audio(r["audio"])
            except Exception:  # noqa: BLE001 - unreadable clip
                continue
            length = len(wav) / SR
            if length > 29.0:
                continue
            feats = processor.feature_extractor(wav, sampling_rate=SR, return_tensors="pt").input_features.to(dev)
            ids = torch.tensor([processor.tokenizer(r["text"]).input_ids], dtype=torch.long, device=dev)
            # the clip's own frame count, jittered so the model sees a range of context lengths around it
            n_ctx = int(round(1500 / 30 * length))
            n_ctx += torch.randint(-min(64, n_ctx // 3), min(64, n_ctx // 3) + 1, (1,)).item()
            n_ctx = max(16, min(1500, n_ctx))
            optimizer.zero_grad()
            partial = model_train.decoder(input_ids=ids, encoder_hidden_states=encode_partial(model_train, feats, n_ctx),
                                          output_hidden_states=True)
            with torch.no_grad():
                full = model_base.decoder(input_ids=ids, encoder_hidden_states=encode_partial(model_base, feats, 1500),
                                          output_hidden_states=True)
            loss = criterion(torch.cat(partial.hidden_states, 0), torch.cat(full.hidden_states, 0))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (i + 1) % 500 == 0:
                print(f"epoch {epoch + 1}/{a.epochs} clip {i + 1} loss {sum(losses[-500:]) / len(losses[-500:]):.5f}", flush=True)
        print(f"epoch {epoch + 1}/{a.epochs} done, mean loss {sum(losses) / max(1, len(losses)):.5f} over {len(losses)} clips", flush=True)

    full_model = WhisperForConditionalGeneration.from_pretrained(a.model)
    full_model.model = model_train.eval().cpu()
    full_model.save_pretrained(a.out)
    processor.save_pretrained(a.out)
    print(f"saved ACFT model -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
