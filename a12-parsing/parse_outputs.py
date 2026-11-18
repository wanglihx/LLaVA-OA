#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone output-parsing script - JMIR revision (Editorial Comment 13).

Extracts discrete KL grades (0-4) from the raw model outputs stored in
4-evalresult, using the regular-expression pattern described in the
manuscript Methods ("Line 162 to Line 175"):

    ^The most likely Kellgren-Lawrence grade of this knee X-ray image is
     Grade (\\d)

Behaviour:
- MLLM outputs (JSON with a "results" key): parse the "predicted_output"
  text with the pattern "Grade\\s*([0-4])". Samples that cannot be parsed
  are reported as parse failures and excluded from downstream metric
  calculation (per the manuscript: instances without a valid prediction are
  treated as invalid data).
- CLIP-family outputs (flat JSON list): labels are already discrete
  ("predicted_label"); no text parsing is needed.
- True labels are read from the same records ("true_grade" for MLLM files,
  "true_label" for CLIP files).

Usage:
    python parse_outputs.py --eval_dir ../4-evalresult --output_dir ./parsed

Outputs (in output_dir):
    parsed_predictions.csv    long-format table: model, image_name,
                              true_grade, predicted_grade, parse_status
    parsing_failures.csv      samples whose outputs could not be parsed
"""
import argparse
import json
import os
import re

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

# Manuscript pattern (Methods): anchored form; the search below falls back to
# a plain "Grade N" match to stay consistent with the agreement and McNemar
# scripts, whose first-match behaviour reproduces the reported numbers.
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


def parse_file(path):
    """Parse one 4-evalresult JSON. Returns (rows, failures)."""
    with open(path, encoding="utf-8-sig") as f:
        raw = json.load(f)

    rows, failures = [], []
    if isinstance(raw, list):
        # CLIP family: labels already discrete
        for it in raw:
            rows.append({
                "image_name": it["image_name"],
                "true_grade": it.get("true_label"),
                "predicted_grade": it.get("predicted_label"),
                "parse_status": "predefined",
            })
    else:
        # MLLM family: parse the generated answer
        for it in raw["results"]:
            text = it.get("predicted_output", "")
            g = parse_grade(text)
            if g is None:
                failures.append({
                    "image_name": it.get("image_name"),
                    "true_grade": it.get("true_grade", it.get("true_label")),
                    "predicted_output": text,
                })
            else:
                rows.append({
                    "image_name": it["image_name"],
                    "true_grade": it.get("true_grade", it.get("true_label")),
                    "predicted_grade": g,
                    "parse_status": "parsed",
                })
    return rows, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_dir", default=os.path.join("..", "4-evalresult"),
                    help="directory of the 4-evalresult model output JSONs")
    ap.add_argument("--output_dir", default=os.path.join(".", "parsed"),
                    help="output directory for parsed CSV files")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    import csv
    all_rows = []
    all_failures = []
    summary = []

    for model, fn in MODELS:
        path = os.path.join(args.eval_dir, fn)
        rows, failures = parse_file(path)
        n_total = len(rows) + len(failures)
        n_parsed = len(rows)
        summary.append((model, n_total, n_parsed, len(failures)))
        for r in rows:
            r["model"] = model
            all_rows.append(r)
        for f in failures:
            f["model"] = model
            all_failures.append(f)
        print("[%s] %s: %d images, %d parsed, %d failed"
              % ("OK" if not failures else "WARN", model, n_total, n_parsed, len(failures)))

    cols = ["model", "image_name", "true_grade", "predicted_grade", "parse_status"]
    with open(os.path.join(args.output_dir, "parsed_predictions.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r[c] for c in cols})

    fcols = ["model", "image_name", "true_grade", "predicted_output"]
    with open(os.path.join(args.output_dir, "parsing_failures.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fcols)
        w.writeheader()
        for r in all_failures:
            w.writerow({c: r.get(c, "") for c in fcols})

    print()
    print("Summary of parse failures:")
    for model, n_total, n_parsed, n_failed in summary:
        print("  %-32s parsed %4d/%4d, failed %d" % (model, n_parsed, n_total, n_failed))
    print()
    print("Outputs written to", os.path.abspath(args.output_dir))
    print("  - parsed_predictions.csv")
    print("  - parsing_failures.csv")


if __name__ == "__main__":
    main()
