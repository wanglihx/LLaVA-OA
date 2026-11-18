# -*- coding: utf-8 -*-
"""
Calculate inter-model agreement matrices (Figure 6 data).

Computes four 16x16 agreement matrices among the ground-truth labels and the
15 evaluated models, matched by image identifier (NOT by row position):

  - Cohen's kappa, binary task (KL grade 0-1 vs 2-4)
  - Linearly weighted Cohen's kappa, multiclass task (KL grades 0-4)
  - Matthews correlation coefficient (MCC), binary task
  - Matthews correlation coefficient (MCC), multiclass task

Usage:
    python calculate_agreement_matrix.py --data_dir /path/to/4-evalresult \
                                         --output_dir /path/to/output

Output (in output_dir):
    figure6_kappa_binary.csv   figure6_kappa_multi.csv
    figure6_mcc_binary.csv     figure6_mcc_multi.csv

Notes:
  - All pairwise comparisons are matched by image_name.
  - Agreement is set to NaN (undefined) when either side predicted a single
    class; such cells are rendered as "NaN" in the figures and explained in
    the figure captions.
  - All four matrices share one common model order: Ground Truth first,
    followed by the 15 models sorted by their linearly weighted multiclass
    kappa against the ground truth (descending, NaN last).
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

# Evaluation output files in 4-evalresult (order = Table 3 of the manuscript)
MODELS = [
    ("Qwen-VL-Max", "000-Qwen-VL-Max.json"),
    ("GPT-4o", "000-GPT-4o.json"),
    ("GPT-5", "000-GPT-5.json"),
    ("Gemini-2.5-Pro", "000-Gemini-2.5-pro.json"),
    ("LLaVA-Med", "000-LLaVA-Med.json"),
    ("CLIP", "000-CLIP.json"),
    ("CLIP-OA", "001-CLIP-OA.json"),
    ("LLaVA-Med-CLIPOA", "002-LLaVA-Med-CLIPOA.json"),
    ("LLaVA-Med-LoRA500", "003-LLaVA-Med-LoRA500.json"),
    ("LLaVA-Med-CLIPOA-LoRA500", "006-LLaVA-Med-CLIPOA-LoRA500.json"),
    ("LLaVA-Med-CLIPOA-MLP500", "007-LLaVA-Med-CLIPOA-MLP500.json"),
    ("LLaVA-Med-CLIPOA-MLP5778", "008-LLaVA-Med-CLIPOA-MLP5778.json"),
    ("LLaVA-Med-CLIPOA-MLP500-LoRA500", "009-LLaVA-Med-CLIPOA-MLP500-LoRA500.json"),
    ("LLaVA-Med-CLIPOA-MLP5778-LoRA500", "011-LLaVA-Med-CLIPOA-MLP5778-LoRA500.json"),
    ("LLaVA-Med-MLP500-LoRA500", "012-LLaVA-Med-MLP500-LoRA500.json"),
]


def parse_grade(text):
    """Extract KL grade (0-4) from a model output string."""
    m = re.search(r"Grade\s*([0-4])", text)
    return int(m.group(1)) if m else None


def load_predictions(path):
    """Load per-image predictions from a 4-evalresult JSON file."""
    with open(path, encoding="utf-8-sig") as f:
        raw = json.load(f)
    if isinstance(raw, list):  # CLIP / CLIP-OA format
        return {it["image_name"]: it["predicted_label"] for it in raw}
    # MLLM format: parse the generated answer
    preds = {}
    for it in raw["results"]:
        g = parse_grade(it.get("predicted_output", ""))
        if g is not None:
            preds[it["image_name"]] = g
    return preds


def to_binary(d):
    """Collapse KL grades into binary (0-1 vs 2-4)."""
    return {k: (0 if v <= 1 else 1) for k, v in d.items()}


def calc_pair(a, b, binary=False, metric="kappa"):
    """Agreement between two prediction maps, matched by image identifier."""
    ca, cb = (to_binary(a), to_binary(b)) if binary else (a, b)
    common = sorted(set(ca) & set(cb))
    ya = [ca[i] for i in common]
    yb = [cb[i] for i in common]
    if len(set(ya)) == 1 or len(set(yb)) == 1:
        return np.nan  # undefined for single-class predictions
    if metric == "kappa":
        return cohen_kappa_score(ya, yb, weights=("linear" if not binary else None))
    return matthews_corrcoef(ya, yb)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default=os.path.join("..", "4-evalresult"),
                    help="path to the 4-evalresult directory")
    ap.add_argument("--output_dir", default=".", help="output directory for CSV files")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    preds = {name: load_predictions(os.path.join(args.data_dir, fn))
             for name, fn in MODELS}
    for name in preds:
        print(f"[OK] {name}: {len(preds[name])} images")

    with open(os.path.join(args.data_dir, "001-CLIP-OA.json"), encoding="utf-8-sig") as f:
        raw_gt = json.load(f)
    gt = {it["image_name"]: it["true_label"] for it in raw_gt}

    all_names = ["Ground Truth"] + [n for n, _ in MODELS]
    maps = {"Ground Truth": gt, **preds}
    n = len(all_names)

    kb = np.full((n, n), np.nan)  # kappa binary
    km = np.full((n, n), np.nan)  # kappa multiclass (linearly weighted)
    mb = np.full((n, n), np.nan)  # MCC binary
    mm = np.full((n, n), np.nan)  # MCC multiclass

    for i in range(n):
        for j in range(n):
            if i == j:
                if len(set(maps[all_names[i]].values())) > 1:
                    kb[i, j] = km[i, j] = mb[i, j] = mm[i, j] = 1.0
                continue  # single-class diagonal stays NaN
            kb[i, j] = calc_pair(maps[all_names[i]], maps[all_names[j]], binary=True, metric="kappa")
            km[i, j] = calc_pair(maps[all_names[i]], maps[all_names[j]], binary=False, metric="kappa")
            mb[i, j] = calc_pair(maps[all_names[i]], maps[all_names[j]], binary=True, metric="mcc")
            mm[i, j] = calc_pair(maps[all_names[i]], maps[all_names[j]], binary=False, metric="mcc")

    # Common order for all four matrices: Ground Truth first, then models by
    # their linearly weighted multiclass kappa vs Ground Truth (descending).
    order = sorted(range(1, n), key=lambda i: (np.isnan(km[i, 0]), -km[i, 0]))
    idx = [0] + order
    names_sorted = [all_names[i] for i in idx]
    print("Common model order:", names_sorted)

    for mat, fn in [(kb, "figure6_kappa_binary.csv"), (km, "figure6_kappa_multi.csv"),
                    (mb, "figure6_mcc_binary.csv"), (mm, "figure6_mcc_multi.csv")]:
        df = pd.DataFrame(mat[idx][:, idx], index=names_sorted, columns=names_sorted)
        df.round(4).to_csv(os.path.join(args.output_dir, fn))
        print(f"[OK] saved {fn}")

    # Sanity check against the manuscript values (Comment 19, 0.894-0.934)
    FIVE = ["LLaVA-Med-CLIPOA-LoRA500", "LLaVA-Med-CLIPOA-MLP500",
            "LLaVA-Med-CLIPOA-MLP5778", "LLaVA-Med-CLIPOA-MLP500-LoRA500",
            "LLaVA-Med-CLIPOA-MLP5778-LoRA500"]
    km_sorted = km[idx][:, idx]
    idx5 = [names_sorted.index(x) for x in FIVE]
    vals = [km_sorted[idx5[a], idx5[b]] for a in range(5) for b in range(a + 1, 5)]
    print(f"Five-configuration pairwise kappa: {min(vals):.4f} - {max(vals):.4f} "
          "(manuscript: 0.894-0.934)")


if __name__ == "__main__":
    main()
