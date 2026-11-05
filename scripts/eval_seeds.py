"""Score each saved categorization seed and the production ensemble on the held-out gold.

The gold file and the seed directories are external and are not part of this
repository. Pass them explicitly:

    python scripts/eval_seeds.py \
        --gold path/to/test.jsonl \
        --seeds path/to/cat_v5c path/to/cat_v5c_s1 path/to/cat_v5c_s2

Each seed is scored with its own ``thresholds.json``. The ensemble averages the
three sigmoid outputs, thresholds with ``config/cascade/thresholds.json`` and
applies the same rule stack as the dashboard. The output is aggregate metrics
only; no sentence text is written.
"""
import argparse
import datetime as dt
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "app", "pipeline"))
sys.path.insert(0, HERE)
import cat_rules  # noqa: E402
import catmodel  # noqa: E402
import pick_seed  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

E5 = "intfloat/multilingual-e5-large-instruct"
INSTR = "Represent the specific topic of this hotel review sentence for clustering."
EXCLUDED = pick_seed.EXCLUDED
DEFAULT_SEEDS = os.path.join(ROOT, "models", "cat_v5c"), os.path.join(ROOT, "models", "cat_v5c_s1"), os.path.join(ROOT, "models", "cat_v5c_s2")


def breakdown(truth, guessed, labels):
    idx = [i for i, label in enumerate(labels) if label not in EXCLUDED]
    per = []
    for k, label in enumerate(labels[i] for i in idx):
        t = [row[k] for row in truth]
        p = [row[k] for row in guessed]
        prec, rec, f1 = pick_seed.prf(t, p)
        per.append({"category": label, "p": round(prec, 4), "r": round(rec, 4), "f1": round(f1, 4)})

    flat_t = [x for row in truth for x in row]
    flat_p = [x for row in guessed for x in row]
    micro = pick_seed.prf(flat_t, flat_p)
    macro = tuple(sum(x[key] for x in per) / len(per) for key in ("p", "r", "f1"))
    return {
        "micro": {"p": round(micro[0], 4), "r": round(micro[1], 4), "f1": round(micro[2], 4)},
        "macro": {"p": round(macro[0], 4), "r": round(macro[1], 4), "f1": round(macro[2], 4)},
        "per_category": per,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=os.getenv("REVIEWS_GOLD"))
    ap.add_argument("--seeds", nargs="+", default=None)
    ap.add_argument("--thresholds", default=os.path.join(ROOT, "config", "cascade", "thresholds.json"))
    ap.add_argument("--rules", help="rule module to import instead of app/pipeline/cat_rules.py")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data", "categorization.json"))
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)

    seeds = a.seeds or [p for p in os.getenv("REVIEWS_SEED_DIRS", "").split(";") if p.strip()]
    if not seeds:
        seeds = [p for p in DEFAULT_SEEDS if os.path.isdir(p)]
    if not seeds:
        raise SystemExit("no seeds: pass --seeds or set REVIEWS_SEED_DIRS")
    if not a.gold:
        raise SystemExit("no gold file: pass --gold or set REVIEWS_GOLD")
    if not os.path.isfile(a.gold):
        raise SystemExit(f"gold file not found: {a.gold}")

    if a.rules:
        spec = importlib.util.spec_from_file_location("bench_rules", a.rules)
        rules = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rules)
    else:
        rules = cat_rules

    examples = pick_seed.load_gold(a.gold)
    labels = catmodel.labels_of(seeds[0])
    dev = torch.device("cpu" if a.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    texts = [text for text, _ in examples]
    idx = [i for i, label in enumerate(labels) if label not in EXCLUDED]
    truth = [[labels[i] in gold for i in idx] for _, gold in examples]
    print(f"gold {len(examples)} examples | {len(idx)} categories | device {dev}")

    methods, probs_list = [], []
    for seed in seeds:
        name = os.path.basename(os.path.normpath(seed))
        probs = catmodel.model_probs(seed, texts, labels, dev=str(dev))
        probs_list.append(probs)
        th = torch.tensor(json.load(open(os.path.join(seed, "thresholds.json"), encoding="utf-8")))
        pred = (probs > th).tolist()
        guessed = [[row[i] for i in idx] for row in pred]
        result = breakdown(truth, guessed, labels)
        result["name"] = name
        methods.append(result)
        print(f"{name:16s} micro F1 {result['micro']['f1']:.3f} | macro F1 {result['macro']['f1']:.3f}")

    probs = torch.stack(probs_list).mean(0).numpy()
    th = np.asarray(json.load(open(a.thresholds, encoding="utf-8")), dtype=np.float32)
    pred = probs > th
    pred = rules.veto_staff(pred, texts)
    pred = rules.apply_rules(pred, probs, texts)

    enc = SentenceTransformer(E5, device=str(dev))
    Xu = enc.encode([f"Instruct: {INSTR}\nQuery: {t}" for t in texts],
                    normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    bath = rules.encode_bathroom(enc, f"Instruct: {INSTR}\nQuery: ")
    pred = rules.water_to_room(pred, Xu, texts, bath)
    del enc, Xu, bath
    torch.cuda.empty_cache()

    guessed = [[bool(row[i]) for i in idx] for row in pred]
    result = breakdown(truth, guessed, labels)
    result["name"] = "ensemble + rules"
    methods.append(result)
    print(f"{result['name']:16s} micro F1 {result['micro']['f1']:.3f} | macro F1 {result['macro']['f1']:.3f}")

    out = {
        "generated": dt.date.today().isoformat(),
        "gold": {"examples": len(examples), "categories": len(idx)},
        "protocol": ("held-out gold; one example per review snippet; a neutral mention counts as "
                     "category-present; Location excluded; per-seed rows use that checkpoint's "
                     "thresholds; the ensemble row averages the three seeds, thresholds with "
                     "config/cascade/thresholds.json and applies the rule stack"),
        "methods": methods,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
