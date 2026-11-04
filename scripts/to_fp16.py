"""One-off: convert the shipped checkpoint to fp16 and report both sizes.

    python scripts/to_fp16.py --model-dir models/cat

The float32 weights are kept next to the converted file as
``model.fp32.safetensors`` so the conversion can be reviewed or repeated.
Neither file is tracked by git; ``models/`` is ignored.
"""
import argparse
import os
import shutil

from safetensors.torch import load_file, save_file


def human(n):
    return f"{n:,} bytes ({n / 1024 / 1024:.1f} MiB)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join("models", "cat"))
    a = ap.parse_args()
    path = os.path.join(a.model_dir, "model.safetensors")
    if not os.path.isfile(path):
        raise SystemExit(f"missing {path}")

    state = load_file(path)
    fp32 = sum(1 for t in state.values() if t.dtype.is_floating_point and t.dtype != __import__("torch").float16)
    if fp32 == 0:
        print(f"{path} is already fp16, {human(os.path.getsize(path))}")
        return

    before = os.path.getsize(path)
    backup = os.path.join(a.model_dir, "model.fp32.safetensors")
    if not os.path.isfile(backup):
        shutil.copy2(path, backup)

    import torch
    state = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in state.items()}
    save_file(state, path, metadata={"format": "pt"})
    after = os.path.getsize(path)
    print(f"fp32 {human(before)}")
    print(f"fp16 {human(after)}")
    print(f"backup {human(os.path.getsize(backup))} -> {backup}")


if __name__ == "__main__":
    main()
