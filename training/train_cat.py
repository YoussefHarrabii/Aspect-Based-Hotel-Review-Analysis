"""Category-only DeBERTa: predict WHICH categories are present, no sentiment.

Reuses the shared data loading in `training/shared.py` (category|polarity labels)
but collapses polarity away.
The label order comes from ``config/taxonomy_v5.json`` and matches the shipped
checkpoint's ``config.json`` ``id2label``.

    python training/train_cat.py --data-dir data/train --out models/cat-trained
"""
import argparse
import collections
import json
import os
import sys
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import shared as T

BASE = os.path.join(ROOT, "models", "deberta-v3-base")
TAX = os.path.join(ROOT, "config", "taxonomy_v5.json")


def collapse(y18, ncat):
    return [[max(row[i], row[ncat + i]) for i in range(ncat)] for row in y18]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "train"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "cat-trained"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--mem-frac", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos-weight-cap", type=float, default=0.0)
    ap.add_argument("--tax", default=TAX)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--opt", default="adamw", choices=("adamw", "adamw8bit"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)

    cats = list(json.load(open(args.tax, encoding="utf-8"))["categories"])
    LAB18 = [f"{c}|positive" for c in cats] + [f"{c}|negative" for c in cats]
    lab2i = {l: i for i, l in enumerate(LAB18)}
    tr_t, tr_y = T.load_examples(os.path.join(args.data_dir, "train.jsonl"), lab2i)
    va_t, va_y = T.load_examples(os.path.join(args.data_dir, "val.jsonl"), lab2i)
    te_t, te_y = T.load_examples(os.path.join(args.data_dir, "test.jsonl"), lab2i)
    tr_y, va_y, te_y = collapse(tr_y, len(cats)), collapse(va_y, len(cats)), collapse(te_y, len(cats))
    print(f"categories {len(cats)} | examples train {len(tr_t)} val {len(va_t)} test {len(te_t)}")

    tok = AutoTokenizer.from_pretrained(args.base)

    def enc(t):
        e = tok(t, truncation=True, max_length=args.max_len, padding="max_length", return_tensors="pt")
        return e["input_ids"], e["attention_mask"]

    ids_tr, m_tr = enc(tr_t); ids_va, m_va = enc(va_t); ids_te, m_te = enc(te_t)
    Ytr, Yva, Yte = torch.tensor(tr_y, dtype=torch.float32), torch.tensor(va_y, dtype=torch.float32), torch.tensor(te_y, dtype=torch.float32)

    cfg = AutoConfig.from_pretrained(args.base)
    cfg.num_labels = len(cats)
    cfg.problem_type = "multi_label_classification"
    cfg.id2label = {i: c for i, c in enumerate(cats)}
    cfg.label2id = {c: i for i, c in enumerate(cats)}
    model = AutoModelForSequenceClassification.from_pretrained(args.base, config=cfg).to(dev)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print("gradient checkpointing ON")
    for n, p in model.named_parameters():
        if "embeddings" in n:
            p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]

    pw = None
    if args.pos_weight_cap > 0:
        pos = Ytr.sum(0).clamp(min=1)
        pw = ((Ytr.shape[0] - pos) / pos).clamp(max=args.pos_weight_cap).to(dev)
        print("pos_weight", [round(float(x), 1) for x in pw])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    if args.opt == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(trainable, lr=args.lr, weight_decay=0.01)
        print("optimizer: AdamW8bit")
    else:
        opt = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    dl = DataLoader(TensorDataset(ids_tr, m_tr, Ytr), batch_size=args.batch, shuffle=True, drop_last=True)
    steps = max(1, len(dl) // args.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
    best_f1, best_th, best_state, t0 = -1.0, None, None, time.time()
    for ep in range(args.epochs):
        model.train(); running = 0.0; opt.zero_grad()
        for i, (bi, bm, by) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss = loss_fn(model(bi.to(dev), bm.to(dev)).logits, by.to(dev))
            (loss / args.accum).backward(); running += loss.item()
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step(); opt.zero_grad()
        pv = T.predict(model, ids_va, m_va)
        th, glob = T.tune_thresholds(pv, Yva, 5)
        p, r, f1 = T.prf(Yva, T.apply_thresholds(pv, th))
        print(f"epoch {ep+1}: loss {running/max(1,len(dl)):.4f} | val micro P {p:.3f} R {r:.3f} F1 {f1:.3f} (th {glob})")
        if f1 > best_f1:
            best_f1, best_th = f1, th
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    os.makedirs(args.out, exist_ok=True)
    if best_state:
        model.load_state_dict(best_state)
    model.save_pretrained(args.out, safe_serialization=True); tok.save_pretrained(args.out)
    json.dump([float(x) for x in best_th], open(os.path.join(args.out, "thresholds.json"), "w"))
    print(f"trained in {(time.time()-t0)/60:.1f} min; best val microF1 {best_f1:.3f}")

    pv = T.predict(model, ids_te, m_te)
    pred = T.apply_thresholds(pv, best_th)
    p, r, f1 = T.prf(Yte, pred)
    print(f"\nTEST category-only ({len(te_t)} units) | micro P {p:.3f} R {r:.3f} F1 {f1:.3f}")
    pr, rc, fv = [], [], []
    for j, c in enumerate(cats):
        n = int(Yte[:, j].sum())
        if n == 0:
            continue
        a, b, d = T.prf(Yte[:, j:j + 1], pred[:, j:j + 1])
        pr.append(a); rc.append(b); fv.append(d)
        print(f"  {c:30s} n={n:4d}  P {a:.2f} R {b:.2f} F1 {d:.2f}")
    print(f"category-macro P {sum(pr)/len(pr):.3f} R {sum(rc)/len(rc):.3f} F1 {sum(fv)/len(fv):.3f}")


if __name__ == "__main__":
    main()
