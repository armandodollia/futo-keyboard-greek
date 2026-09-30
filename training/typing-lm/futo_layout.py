"""Print the `layout:` block of a language config from a FUTO layout YAML (futo-org/futo-keyboard-layouts).

Usage: python futo_layout.py LatinScript/albanian.yaml          (path in that repo, fetched from GitHub)
       python futo_layout.py https://raw.githubusercontent.com/.../greek.yaml
       python futo_layout.py my_checkout/Greek/greek.yaml

A plain tap types the first character of a key; the rest are long-press alternates. FUTO centres every row, so the
x offset of a row is (keys in the widest row - keys in this row) / 2. Custom key widths are not handled: check them.
"""
import pathlib
import sys
import unicodedata
import urllib.request

import yaml

RAW = "https://raw.githubusercontent.com/futo-org/futo-keyboard-layouts/main/"


def base_key(entry):
    """First character a key types (entry forms: 'q', ['ε', 'έ', ...], {type: case, normal: [...]}, {spec: 'ς'})."""
    if isinstance(entry, list):
        return str(entry[0]), [str(a) for a in entry[1:] if a != "%"]
    if isinstance(entry, dict):
        for k in ("normal", "spec", "base"):
            if k in entry:
                return base_key(entry[k] if isinstance(entry[k], list) else [entry[k]])
        return None, []
    return str(entry), []


def main(src):
    if pathlib.Path(src).exists():
        text = pathlib.Path(src).read_text(encoding="utf-8")
    else:
        text = urllib.request.urlopen(src if src.startswith("http") else RAW + src).read().decode("utf-8")
    lay = yaml.safe_load(text)
    rows, longpress = [], {}
    for row in lay.get("rows", []):
        keys = row.get("letters") if isinstance(row, dict) else None
        if keys is None:
            continue
        if isinstance(keys, str):
            keys = keys.split()
        letters = []
        for e in keys:
            b, alts = base_key(e)
            if b:
                letters.append(b)
                if alts:
                    longpress[b] = alts
        rows.append(letters)
    widest = max(len(r) for r in rows)
    own = sorted({c for r in rows for c in r if c.isalpha() and len(unicodedata.normalize("NFD", c)) > 1})
    print(f"# from {src} ({lay.get('name', '?')}, languages: {lay.get('languages', '?')})")
    print("layout:")
    print(f"  futo: {src}")
    print("  rows:")
    for r in rows:
        print(f"    - ['{''.join(r)}', {(widest - len(r)) / 2}]")
    print(f"  own_keys: '{''.join(own)}'")
    if longpress:
        print("# long-press alternates (typed as the base letter by a normal tap):")
        for b, alts in longpress.items():
            print(f"#   {b}: {' '.join(alts)}")


if __name__ == "__main__":
    main(sys.argv[1])
