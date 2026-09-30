"""Quality/speed check of an exported GGUF through llama-server's raw /completion endpoint, with the exact prompt
string the app sends (common.raw_prompt), on each language's hard cases. Start the server first, e.g.
  llama-server -m cleanup-q4_0.gguf --port 8099            (or add --lora cleanup-el-lora.gguf)
Usage: python q4_check.py --url http://127.0.0.1:8099 --langs el,sq,en
"""
import argparse
import json
import time
import urllib.request

from common import load_lang, raw_prompt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--langs", default="el,sq,en")
    a = ap.parse_args()
    for code in a.langs.split(","):
        for mode, text in load_lang(code).hard_cases:
            body = {"prompt": raw_prompt(mode, code, text), "n_predict": int(len(text) * 0.8) + 40, "temperature": 0,
                    "stop": ["<|im_end|>"], "cache_prompt": False}
            t0 = time.time()
            req = urllib.request.Request(a.url.rstrip("/") + "/completion", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=120))
            t = r["timings"]
            print(f"[{code}/{mode}] {time.time() - t0:.2f}s  prompt {t['prompt_n']} tok @ {t['prompt_per_second']:.0f}/s, "
                  f"gen {t['predicted_n']} tok @ {t['predicted_per_second']:.0f}/s\n   {text}\n-> {r['content'].strip()}",
                  flush=True)


if __name__ == "__main__":
    main()
