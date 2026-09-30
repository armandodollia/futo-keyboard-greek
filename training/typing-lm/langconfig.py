"""Per-language settings (languages/<code>.yaml) and the keystroke simulator shared by prepare/train/evaluate.

Everything language-specific lives in the YAML file: alphabet, keyboard layout, which diacritics have their own key,
typing-error rules, data sources and mix weights. See languages/template.yaml for every key, documented.

The simulator mirrors what FUTO Keyboard feeds the model in PredictCorrection (with patch 0001): one input position
per keystroke, whose embedding is the weighted mix of the embeddings of the (up to) 4 nearest LETTER keys
(char_embed_mixing_v1). Non-letter keys (e.g. Greek ";") count for the geometry but never get weight.
"""
import pathlib
import re
import sys
import unicodedata

import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
for _s in (sys.stdout, sys.stderr):  # pieces like "▁" and non-Latin text crash a cp1252 Windows console / pipe
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DEFAULTS = {
    "min_script_share": 0.9,
    "sentence_end": ".!?…",
    "tokenizer": {"vocab": 16000, "sample_lines": 1_500_000},
    "tap_errors": {"drop": 0.08, "extra": 0.06, "swap": 0.05, "double": 0.03, "partial": 0.35},
    "typing_errors": [],
    "eval": {},
}


def load(lang):
    """lang: a language code (-> languages/<code>.yaml) or a path to a YAML file."""
    path = pathlib.Path(lang)
    if not path.suffix:
        path = HERE / "languages" / f"{lang}.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for k, v in DEFAULTS.items():
        cfg[k] = {**v, **cfg.get(k, {})} if isinstance(v, dict) else cfg.get(k, v)
    cfg["_path"] = str(path.resolve())
    cfg["_dir"] = cfg.get("config_dir") or str(path.resolve().parent)  # set in a trained model's language.yaml copy
    for need in ("code", "name", "letters", "layout", "sources"):
        assert need in cfg, f"{path}: missing '{need}'"
    return cfg


class Language:
    def __init__(self, cfg, tags=()):
        self.cfg, self.tags = cfg, set(tags)
        self.word_re = re.compile(f"[{cfg['letters']}]+")
        # key centres in key units: row r, key i -> (x offset + i + 0.5, r + 0.5)
        self.keypos = {c: (float(x0) + i + 0.5, r + 0.5)
                       for r, (keys, x0) in enumerate(cfg["layout"]["rows"]) for i, c in enumerate(keys)}
        self.letter_keys = [c for c in self.keypos if c.isalpha()]
        self.key_xy = np.array([self.keypos[c] for c in self.letter_keys], dtype=np.float32)
        self.neighbours = {c: [self.letter_keys[j] for j in np.argsort(((self.key_xy - self.keypos[c]) ** 2).sum(1))[1:5]]
                           for c in self.letter_keys}
        self.own_keys = set(cfg["layout"].get("own_keys", ""))
        self.key_tok = None

    def bind(self, sp):
        """Token id of each letter key (its single-letter piece, as the patched keyboard looks it up)."""
        self.key_tok = np.array([sp.piece_to_id(c) for c in self.letter_keys], dtype=np.int64)
        missing = [c for c, t in zip(self.letter_keys, self.key_tok) if t == sp.unk_id()]
        assert not missing, f"keyboard letters without a single-letter piece: {missing}"
        return self

    def is_word(self, w):
        return self.word_re.fullmatch(w) is not None

    def base_letters(self, word):
        """What the keys under the finger were: lowercase, own-key letters kept, every other diacritic removed
        (it is a long-press alternate, so a normal tap gives the base letter and the model restores the accent)."""
        out = []
        for c in word.lower():
            if c in self.own_keys:
                out.append(c)
            else:
                out.extend(ch for ch in unicodedata.normalize("NFD", c) if unicodedata.category(ch) != "Mn")
        return "".join(out)

    # ------------------------------------------------------------ typing errors (config: typing_errors)
    def _rule(self, rule, k, rng):
        """Apply one rule to the key string k; None when it does not apply to this word."""
        if len(k) < rule.get("min_len", 0):
            return None
        if "replace_all" in rule:                        # e.g. typed without diacritics: ë->e, ç->c everywhere
            m = rule["replace_all"]
            return "".join(m.get(c, c) for c in k) if any(c in m for c in k) else None
        if "drop_final" in rule:                         # dropped final letter (Albanian final ë / vowel)
            return k[:-1] if k and k[-1] in rule["drop_final"] else None
        if "final" in rule:                              # last letter replaced (Greek ς typed as σ)
            return k[:-1] + rule["final"][k[-1]] if k and k[-1] in rule["final"] else None
        if "replace_one" in rule:                        # one occurrence replaced by one of the options (q/ç/c)
            m = rule["replace_one"]
            pos = [i for i, c in enumerate(k) if c in m]
            if not pos:
                return None
            i = rng.choice(pos)
            return k[:i] + rng.choice(m[k[i]]) + k[i + 1:]
        if "substitute" in rule:                         # multi-letter misspellings: ει->ι, sh->s, rr->r ...
            start = rule.get("from", 0)
            if rule.get("pick", "present") == "any":     # any pair; no-op if it doesn't occur (as the Greek v1)
                a, b = rng.choice(rule["substitute"])
            else:
                opts = [(a, b) for a, b in rule["substitute"] if k.find(a, start) >= 0]
                if not opts:
                    return None
                a, b = rng.choice(opts)
            hits = [m for m in range(start, len(k)) if k.startswith(a, m)]
            if not hits:
                return k
            i = hits[0] if rule.get("occurrence", "random") == "first" else rng.choice(hits)
            return k[:i] + b + k[i + len(a):]
        if "ending" in rule:                             # wrong inflection ending: the context has to pick the right one
            grp = [(g, e) for g in rule["ending"] for e in sorted(g, key=len, reverse=True) if k.endswith(e)]
            if not grp:
                return None
            g, e = grp[0]
            return k[:-len(e)] + rng.choice([x for x in g if x != e])
        raise ValueError(f"unknown typing_errors rule: {rule}")

    def _stage(self, stage, k, rng):
        if stage.get("tag") and stage["tag"] not in self.tags:
            return k
        if len(k) < stage.get("min_len", 0):
            return k
        if "one_of" in stage:                            # exclusive: one draw, cumulative probabilities
            r, cum = rng.random(), 0.0
            for rule in stage["one_of"]:
                cum += rule["p"]
                if r < cum:
                    out = self._rule(rule, k, rng)
                    if out is not None:
                        return out
                    if not rule.get("fallthrough"):      # doesn't fit this word: nothing happens, unless it passes
                        return k                         # its share on to the next rule
            return k
        if rng.random() < stage["p"]:
            out = self._rule(stage, k, rng)
            return k if out is None else out
        return k

    def keystrokes(self, word, rng):
        keys = self.base_letters(word)
        for stage in self.cfg["typing_errors"]:
            keys = self._stage(stage, keys, rng)
        te, keys, r = self.cfg["tap_errors"], list(keys), rng.random()
        c1, c2, c3, c4 = np.cumsum([te["drop"], te["extra"], te["swap"], te["double"]])
        if len(keys) > 3 and r < c1:                     # dropped key (never the first)
            del keys[rng.randrange(1, len(keys))]
        elif len(keys) > 2 and r < c2:                   # extra (neighbouring) key
            i = rng.randrange(len(keys)); keys.insert(i + 1, rng.choice(self.neighbours.get(keys[i], [keys[i]])))
        elif len(keys) > 3 and r < c3:                   # swapped keys
            i = rng.randrange(len(keys) - 1); keys[i], keys[i + 1] = keys[i + 1], keys[i]
        elif len(keys) > 2 and r < c4:                   # doubled key
            i = rng.randrange(len(keys)); keys.insert(i, keys[i])
        if len(keys) > 2 and rng.random() < te["partial"]:  # partial word -> completion
            keys = keys[:rng.randrange(1, len(keys))]
        return [k for k in keys if k in self.neighbours]

    def tap_mix(self, key, sigma, rng):
        """FUTO-style token mix for one tap: up to 4 letter keys, weights by distance to a noisy tap point."""
        x, y = self.keypos[key]
        tx, ty = x + rng.gauss(0, sigma), y + rng.gauss(0, sigma * 0.8)
        d2 = ((self.key_xy - (tx, ty)) ** 2).sum(1)
        near = np.argsort(d2)[:4]
        w = np.exp(-d2[near] / (2 * 0.35 ** 2))
        w = w / w.sum(); w[w < 0.05] = 0; w = w / w.sum()
        return self.key_tok[near], w.astype(np.float32)
