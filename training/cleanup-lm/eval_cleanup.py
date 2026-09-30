"""Evaluate cleanup models per language on held-out data (work/val_pairs.json, never trained on):
  similarity   1 - CER against the target
  swears       swear-word retention (the language's stems + English)
  language     output is still in the target language (config "detect" regex)
  corrections  held-out trailing self-corrections (augment_corrections.py generator, fixed seed) built from the
               validation messages: 1 - CER against the original sentence, case-insensitive
--hard also prints each language's hand-made hard cases.
MODEL is a merged model folder, an HF id, or a LoRA adapter folder (on top of --base). "name=path" names it; if the
name is a language code, that model is scored on that language only (per-language specialists vs one shared model):
Usage: python eval_cleanup.py --work work shared=work/out/cleanup el=work/out/cleanup-el-adapter [--hard]
"""
import argparse
import json
import os

from augment_corrections import corrector_for, load_messages
from common import kept_language, load_lang, load_model, swears


def correction_items(work, val, per_lang, seed=123):
    c = corrector_for(load_messages(work), seed)
    items, by = [], {}
    for r in (r for r in val if r["mode"] == "light" and r["lang"] in c.langs):
        for _ in range(5):
            s = c.make(r["output"], r["lang"])
            if s:
                if by.get(r["lang"], 0) < per_lang:
                    by[r["lang"]] = by.get(r["lang"], 0) + 1; items.append((r["lang"], s, r["output"]))
                break
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--work", default="work")
    ap.add_argument("--langs", default="", help="languages to score (default: all in val_pairs.json)")
    ap.add_argument("--n", type=int, default=0, help="max validation pairs per language (0 = all)")
    ap.add_argument("--corrections", type=int, default=50, help="correction items per language (0 = skip)")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B", help="base model for adapter folders")
    ap.add_argument("--hard", action="store_true")
    a = ap.parse_args()

    import jiwer
    val = json.load(open(os.path.join(a.work, "val_pairs.json"), encoding="utf-8"))
    codes = [c for c in (a.langs.split(",") if a.langs else sorted({r["lang"] for r in val})) if c]
    langs = {c: load_lang(c) for c in codes}
    en = [load_lang("en")] if os.path.exists(os.path.join(os.path.dirname(__file__), "languages", "en.json")) else []
    corr = correction_items(a.work, val, a.corrections) if a.corrections else []
    sim = lambda ref, out: 1 - min(1.0, jiwer.cer(ref, out) if ref else 1.0)  # noqa: E731

    results = {}
    for spec in a.models:
        name, path = spec.split("=", 1) if "=" in spec else (spec, spec)
        run, model = load_model(path, a.base)
        res = results[name] = {}
        for code in ([name] if name in langs else codes):
            lang = langs[code]
            sw_langs = [lang] + [e for e in en if e.code != code]
            items = [r for r in val if r["lang"] == code][:a.n or None]
            s, sw_in, sw_out, ok = [], 0, 0, 0
            for r in items:
                o = run(r["mode"], code, r["input"])
                s.append(sim(r["output"], o))
                sw_in += swears(r["input"], sw_langs); sw_out += min(swears(o, sw_langs), swears(r["input"], sw_langs))
                ok += kept_language(o, r["output"], lang)
            cs = [sim(orig.lower(), run("light", l, t).lower()) for l, t, orig in corr if l == code]
            res[code] = {"similarity": 100 * sum(s) / max(1, len(s)), "swears": 100 * sw_out / max(1, sw_in),
                         "swears_in_val": sw_in, "language": 100 * ok / max(1, len(items)),
                         "corrections": 100 * sum(cs) / len(cs) if cs else None, "n": len(items)}
            x = res[code]
            print(f"{name} [{code}]: similarity {x['similarity']:.1f}%  swears {x['swears']:.1f}% (of {sw_in})  "
                  f"language {x['language']:.1f}%" + (f"  corrections {x['corrections']:.1f}%" if cs else "")
                  + f"  ({len(items)} pairs)", flush=True)
            if a.hard:
                for mode, text in lang.hard_cases:
                    print(f"  [{code}/{mode}] {text}\n     -> {run(mode, code, text)}", flush=True)
        del model
    out = os.path.join(a.work, "eval_results.json")
    json.dump(results, open(out, "w", encoding="utf-8"), indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
