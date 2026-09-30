"""Build dictation-cleanup training pairs for one language.

For every clean source message:
  input   = the message turned into messy dictation by rule (common.messy: lowercase, no punctuation, fillers,
            repeated words, false starts, a self-correction); 15% of items stay clean, so the model learns to leave
            good text alone
  light   = the ORIGINAL message (perfect by construction: same words, swearing intact, correction applied)
  rambler = a teacher LLM's concise rewrite of the original, kept only if it passes the same kind of checks as the
            keyboard: swear words kept, sane length, no chatter ("Here is...")
Output: jsonl rows {lang, mode, input, output}.

Teacher: any OpenAI-compatible chat server (llama-server, vLLM, Ollama, LM Studio, a hosted API), or MLX on a Mac.
Pick a model that is fluent in the language and does not censor; read a few dozen rewrites before a full run.
Usage: python gen_pairs.py --lang el --sources work/sources_el.txt --out work/pairs_el.jsonl \
         --teacher openai --url http://localhost:8080/v1 [--model NAME] [--n 8000]
       python gen_pairs.py --lang el ... --teacher mlx --model <mlx model>
The config's optional "no_add_words" rejects rewrites that add one of those words (slang, another dialect's words)
that the message does not have. The config's optional "teacher_note" is appended to the teacher's instruction (e.g. to keep a dialect).
--wait-cmd runs a shell command before every chunk of 60 messages and waits while it fails, e.g. to pause while a
shared server is serving a different model.
"""
import argparse
import json
import os
import random
import re
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from common import added_words, load_lang, messy, swears, write_jsonl

RAMBLER = ("Rewrite this {lang} message as a short, clear message the speaker could send as-is. Keep the meaning, "
           "tone, attitude and first-person voice; remove repetition and tangents, but never make it more polite. "
           "Keep {lang}: never translate. Keep slang, insults, threats, obscenities and swear words exactly; never "
           "soften, censor or euphemize. If the message is already short and clear, return it unchanged. Output only "
           "the message.")


def clean_reply(t):
    return re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip().strip('"')


class OpenAITeacher:
    def __init__(self, a):
        self.url = a.url.rstrip("/") + "/chat/completions"
        self.model, self.key, self.no_think = a.model or "default", a.api_key, a.no_think
        self.pool = ThreadPoolExecutor(a.workers)

    def one(self, msg, system):
        body = {"model": self.model, "temperature": 0.3, "max_tokens": 200,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": msg}]}
        if self.no_think:   # llama-server / vLLM: switch off Qwen3-style thinking
            body["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
        for _ in range(3):
            try:
                return clean_reply(json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"])
            except Exception:  # noqa: BLE001  (busy server, timeout: retry, then give up on this item)
                time.sleep(5)
        return ""

    def __call__(self, msgs, system):
        return list(self.pool.map(lambda m: self.one(m, system), msgs))


class MLXTeacher:
    def __init__(self, a):
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
        self.model, self.tok = load(a.model)
        self.generate, self.sampler = generate, make_sampler(temp=0.3)
        self.kw = {"enable_thinking": False} if "qwen3" in a.model.lower() else {}

    def __call__(self, msgs, system):
        out = []
        for m in msgs:
            chat = [{"role": "system", "content": system}, {"role": "user", "content": m}]
            prompt = self.tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, **self.kw)
            out.append(clean_reply(self.generate(self.model, self.tok, prompt, max_tokens=160, sampler=self.sampler)))
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--sources", required=True, help="clean messages, one per line (sample_sources.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8000, help="max messages to use")
    ap.add_argument("--teacher", choices=["openai", "mlx"], default="openai")
    ap.add_argument("--url", default="http://localhost:8080/v1", help="OpenAI-compatible base URL")
    ap.add_argument("--model", default="", help="model name on the server, or the MLX model")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="default: $OPENAI_API_KEY")
    ap.add_argument("--no-think", action="store_true", help="send chat_template_kwargs.enable_thinking=false")
    ap.add_argument("--workers", type=int, default=6, help="parallel requests (match the server's slots)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wait-cmd", default="", help="shell command run before each chunk; wait while it exits non-zero")
    a = ap.parse_args()

    lang = load_lang(a.lang)
    try:   # English swearing shows up in every language, so it counts too
        swear_langs = [lang] + ([load_lang("en")] if lang.code != "en" else [])
    except SystemExit:
        swear_langs = [lang]
    teacher = OpenAITeacher(a) if a.teacher == "openai" else MLXTeacher(a)
    system = RAMBLER.format(lang=lang.name) + (" " + lang.teacher_note if lang.teacher_note else "")
    msgs = [l.strip() for l in open(a.sources, encoding="utf-8") if l.strip()][:a.n]
    rng = random.Random(a.seed)
    n_ok = n_bad = 0
    t0 = time.time()
    with open(a.out, "w", encoding="utf-8") as out:
        for s in range(0, len(msgs), 60):
            chunk = msgs[s:s + 60]
            while a.wait_cmd and subprocess.run(a.wait_cmd, shell=True).returncode != 0:
                print("wait-cmd failed, waiting 60 s", flush=True); time.sleep(60)
            for msg, ram in zip(chunk, teacher(chunk, system)):
                inp = msg if rng.random() < 0.15 else messy(msg, lang, rng)
                write_jsonl(out, {"lang": lang.code, "mode": "light", "input": inp, "output": msg})
                ratio = len(ram) / max(1, len(msg))
                if (ram and 0.2 <= ratio <= 1.3 and swears(ram, swear_langs) * 2 >= swears(msg, swear_langs)
                        and not ram.lower().startswith(("here", "sure", "i can")) and not added_words(msg, ram, lang)):
                    write_jsonl(out, {"lang": lang.code, "mode": "rambler", "input": inp, "output": ram}); n_ok += 1
                else:
                    n_bad += 1
            out.flush()
            print(f"{lang.code} {s + len(chunk)}/{len(msgs)}  rambler kept {n_ok}, rejected {n_bad}  "
                  f"{(time.time() - t0) / 60:.0f} min", flush=True)
    print("done", n_ok, n_bad)


if __name__ == "__main__":
    main()
