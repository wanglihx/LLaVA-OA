#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metric calculation script - JMIR revision (Editorial Comment 13).

Computes the classification metrics reported in Table 3 of the manuscript
for the 15 evaluated models, from the raw evaluation outputs in 4-evalresult.

Parsing is identical to a12-parsing/parse_outputs.py: KL grades are extracted
with the manuscript pattern (anchored form, falling back to a plain
"Grade N" match). Samples whose outputs cannot be parsed are excluded from
metric calculation, matching the exclusion of 7 Gemini-2.5-Pro outputs
described in the manuscript.

Metrics:
  - Multiclass (KL grades 0-4): accuracy, macro precision/recall/F1,
    weighted precision/recall/F1, mean absolute error (MAE, equal-interval
    assumption on the ordinal KL scale).
  - Binary (KL 0-1 vs 2-4, cutoff at KL=2): accuracy, precision,
    sensitivity (recall), specificity, F1.
  - Per-class (KL 0-4): sample count, accuracy (recall), sensitivity,
    precision, F1, and Wilson 95% confidence intervals for per-class
    accuracy.

Usage:
    python metric_calculation.py --eval_dir ../4-evalresult --output_dir ./metrics

Outputs (in output_dir):
    multiclass_metrics.csv     one row per model
    binary_metrics.csv         one row per model
    per_class_metrics.csv      one row per model per KL grade
"""
import argparse
import csv
import json
import os
import re

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

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

BINARY_CUTOFF = 2  # KL 0-1 vs KL 2-4

ANCHORED_PATTERN = re.compile(
    r"The most likely Kellgren-Lawrence grade of this knee X-?ray image is Grade\s*([0-4])",
    re.IGNORECASE,
)
PLAIN_PATTERN = re.compile(r"Grade\s*([0-4])")


def parse_grade(text):
    """Extract KL grade (0-4) from a model output string, or None on failure."""
    if text is None:
        return None
    m = ANCHORED_PATTERN.search(text)
    if m is None:
        m = PLAIN_PATTERN.search(text)
    return int(m.group(1)) if m else None


def load_predictions(path):
    """Load per-image (true, pred) pairs from a 4-evalresult JSON.

    Unparsable MLLM outputs are skipped (excluded from metric calculation).
    """
    with open(path, encoding="utf-8-sig") as f:
        raw = json.load(f)

    y_true, y_pred, n_failed = [], [], 0
    if isinstance(raw, list):
        # CLIP family: labels already discrete
        for it in raw:
            y_true.append(it["true_label"])
            y_pred.append(it["predicted_label"])
    else:
        # MLLM family: parse the generated answer
        for it in raw["results"]:
            g = parse_grade(it.get("predicted_output", ""))
            if g is None:
                n_failed += 1
                continue
            y_true.append(it.get("true_grade", it.get("true_label")))
            y_pred.append(g)
    return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), n_failed


def to_binary(y):
    return (y >= BINARY_CUTOFF).astype(int)


def specificity(y_true_bin, y_pred_bin):
    """Binary specificity from the confusion matrix."""
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def mae(y_true, y_pred):
    """Mean absolute error under the equal-interval assumption."""
    return float(np.mean(np.abs(y_true.astype(float) - y_pred.astype(float))))


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return centre - half, centre + half


def multiclass_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "mae": mae(y_true, y_pred),
    }


def binary_metrics(y_true, y_pred):
    yt, yp = to_binary(y_true), to_binary(y_pred)
    return {
        "accuracy": accuracy_score(yt, yp) * 100,
        "precision": precision_score(yt, yp, average="binary", zero_division=0) * 100,
        "sensitivity": recall_score(yt, yp, average="binary", zero_division=0) * 100,
        "specificity": specificity(yt, yp) * 100 if not np.isnan(specificity(yt, yp)) else np.nan,
        "f1": f1_score(yt, yp, average="binary", zero_division=0) * 100,
    }


def per_class_metrics(y_true, y_pred):
    rows = []
    for k in range(5):
        idx = np.where(y_true == k)[0]
        n = len(idx)
        if n == 0:
            continue
        yt, yp = y_true[idx], y_pred[idx]
        n_correct = int(np.sum(yt == yp))
        lo, hi = wilson_ci(n_correct, n)
        rows.append({
            "kl_grade": k,
            "n": n,
            "accuracy": n_correct / n * 100,
            "accuracy_ci95_low": lo * 100,
            "accuracy_ci95_high": hi * 100,
            "sensitivity": recall_score(y_true, y_pred, labels=[k], average=None,
                                        zero_division=0)[0] * 100,
            "precision": precision_score(y_true, y_pred, labels=[k], average=None,
                                         zero_division=0)[0] * 100,
            "f1": f1_score(y_true, y_pred, labels=[k], average=None, zero_division=0)[0] * 100,
        })
    return rows


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in fieldnames})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_dir", default=os.path.join("..", "4-evalresult"),
                    help="directory of the 4-evalresult model output JSONs")
    ap.add_argument("--output_dir", default=os.path.join(".", "metrics"),
                    help="output directory for metric CSV files")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    multi_rows, binary_rows, per_class_rows = [], [], []
    for model, fn in MODELS:
        y_true, y_pred, n_failed = load_predictions(os.path.join(args.eval_dir, fn))
        if len(y_true) == 0:
            print("[FAIL] %s: no usable samples" % model)
            continue
        multi = multiclass_metrics(y_true, y_pred)
        bina = binary_metrics(y_true, y_pred)
        multi_rows.append({"model": model, "n_images": len(y_true), "n_excluded": n_failed, **multi})
        binary_rows.append({"model": model, "n_images": len(y_true), "n_excluded": n_failed, **bina})
        for r in per_class_metrics(y_true, y_pred):
            per_class_rows.append({"model": model, **r})
        print("[%s] %s: %d images, %d excluded; acc=%.2f%% f1_macro=%.2f%% MAE=%.3f"
              % ("WARN" if n_failed else "OK", model, len(y_true), n_failed,
                 multi["accuracy"], multi["f1_macro"], multi["mae"]))

    multi_fields = ["model", "n_images", "n_excluded", "accuracy",
                    "precision_macro", "recall_macro", "f1_macro",
                    "precision_weighted", "recall_weighted", "f1_weighted", "mae"]
    binary_fields = ["model", "n_images", "n_excluded", "accuracy",
                     "precision", "sensitivity", "specificity", "f1"]
    per_class_fields = ["model", "kl_grade", "n", "accuracy",
                        "accuracy_ci95_low", "accuracy_ci95_high",
                        "sensitivity", "precision", "f1"]

    write_csv(os.path.join(args.output_dir, "multiclass_metrics.csv"), multi_fields, multi_rows)
    write_csv(os.path.join(args.output_dir, "binary_metrics.csv"), binary_fields, binary_rows)
    write_csv(os.path.join(args.output_dir, "per_class_metrics.csv"), per_class_fields, per_class_rows)

    print()
    print("Outputs written to", os.path.abspath(args.output_dir))
    print("  - multiclass_metrics.csv")
    print("  - binary_metrics.csv")
    print("  - per_class_metrics.csv")


if __name__ == "__main__":
    main()
