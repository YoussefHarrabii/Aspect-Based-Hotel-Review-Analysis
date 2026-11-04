"""Final segmenter: local, pretrained, no training.

    slice (>~200w) -> squash !!/?? -> strip ALL punctuation -> restore -> split on "."

The restorer (deepmultilingualpunctuation) decides the boundaries; we just trust its periods.
The restorer normalizes !? itself, so there is nothing to preserve on the way in.

    python app/pipeline/restore_seg.py            # selftest
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repo root (app/pipeline -> root)
sys.path.insert(0, HERE)

import transformers.pipelines.token_classification as TC
_orig = TC.TokenClassificationPipeline._sanitize_parameters
TC.TokenClassificationPipeline._sanitize_parameters = \
    lambda self, **k: (k.pop("grouped_entities", None), _orig(self, **k))[1]
from deepmultilingualpunctuation import PunctuationModel

BOUND = re.compile(r"[.!?]")
_PM = None
PUNCT_HUB = "oliverguhr/fullstop-punctuation-multilang-large"


def _pm():
    global _PM
    if _PM is None:
        # installer snapshot when present (pinned revision + `.complete` sentinel),
        # else the hub id (dev machines)
        local = os.path.join(ROOT, "models", "punct")
        src = local if os.path.isfile(os.path.join(local, ".complete")) else PUNCT_HUB
        _PM = PunctuationModel(src)
    return _PM


def slice_blocks(text, target=200, radius=30):
    """Cut into <= target+radius word blocks at the nearest sentence punctuation (else target)."""
    ws = (text or "").split()
    if len(ws) <= target:
        return [text or ""]
    cuts, i = [0], 0
    while len(ws) - i > target:
        lo, hi = max(1, target - radius), min(len(ws) - i - 1, target + radius)
        best = None
        for j in range(lo, hi + 1):
            if BOUND.search(ws[i + j - 1]) and (best is None or abs(j - target) < abs(best - target)):
                best = j
        i += best if best else target
        cuts.append(i)
    cuts.append(len(ws))
    return [" ".join(ws[a:b]) for a, b in zip(cuts, cuts[1:]) if b > a]


def units(text):
    out = []
    for block in slice_blocks(text):
        c = re.sub(r"!{2,}", "!", block)
        c = re.sub(r"\?{2,}", "?", c)
        c = re.sub(r"\s+", " ", re.sub(r"[.,!?;:]+", " ", c)).strip()
        if not c:
            continue
        restored = _pm().restore_punctuation(c)
        out += [s.strip() for s in re.split(r"(?<=\.)\s+", restored) if s.strip()]
    return out


def selftest():
    burst = "The restaurant was so bad! No curry? No water? No salt? No spice? No music?"
    u = units(burst)
    assert len(u) == 1, u
    two = "The pool was great. The restaurant was awful."
    assert len(units(two)) == 2, units(two)
    long_txt = " ".join(["word"] * 700)
    assert all(len(b.split()) <= 230 for b in slice_blocks(long_txt))
    print("selftest OK")
    print("burst ->", units(burst))
    print("two   ->", units(two))


if __name__ == "__main__":
    selftest()
