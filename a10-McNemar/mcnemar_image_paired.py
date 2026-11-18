#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
McNemar image-paired recomputation - JMIR revision round (2026-11-15)
- Pairs predictions by image_name (not row position)
- Computes both uncorrected and continuity-corrected McNemar chi-square
- Holm correction applied within each comparison family
- Optionally compares against original row-position results (Group A)

Usage:
    python mcnemar_image_paired.py \
        --eval_dir ../4-evalresult \
        --clipoa_json ../4-evalresult/001-CLIP-OA.json \
        --focal_json ../8-clip-loss/evaluation_results_clip_focal.json \
        --weighted_json ../8-clip-loss/evaluation_results_detailed_weighted.json \
        --output_dir ./mcnemar_results \
        [--orig55_csv /path/to/mcnemar_test_results.csv] \
        [--orig3_csv /path/to/mcnemar_test_three_models.csv]
"""
import argparse
import json
import os
import re
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import chi2

# ---------- Model lists (same as original manuscript) ----------
MODELS55 = [
    '000-CLIP.json',
    '001-CLIP-OA.json',
    '000-LLaVA-Med.json',
    '002-LLaVA-Med-CLIPOA.json',
    '003-LLaVA-Med-LoRA500.json',
    '006-LLaVA-Med-CLIPOA-LoRA500.json',
    '007-LLaVA-Med-CLIPOA-MLP500.json',
    '008-LLaVA-Med-CLIPOA-MLP5778.json',
    '009-LLaVA-Med-CLIPOA-MLP500-LoRA500.json',
    '011-LLaVA-Med-CLIPOA-MLP5778-LoRA500.json',
    '012-LLaVA-Med-MLP500-LoRA500.json',
]
NAMES55 = {
    '000-CLIP.json': 'CLIP',
    '001-CLIP-OA.json': 'CLIP-OA',
    '000-LLaVA-Med.json': 'LLaVA-Med',
    '002-LLaVA-Med-CLIPOA.json': 'LLaVA+CLIP-OA',
    '003-LLaVA-Med-LoRA500.json': 'LLaVA+LoRA500',
    '006-LLaVA-Med-CLIPOA-LoRA500.json': 'LLaVA+CLIP-OA+LoRA500',
    '007-LLaVA-Med-CLIPOA-MLP500.json': 'LLaVA+CLIP-OA+MLP500',
    '008-LLaVA-Med-CLIPOA-MLP5778.json': 'LLaVA+CLIP-OA+MLP5778',
    '009-LLaVA-Med-CLIPOA-MLP500-LoRA500.json': 'LLaVA+CLIP-OA+MLP500+LoRA500',
    '011-LLaVA-Med-CLIPOA-MLP5778-LoRA500.json': 'LLaVA+CLIP-OA+MLP5778+LoRA500',
    '012-LLaVA-Med-MLP500-LoRA500.json': 'LLaVA+MLP500+LoRA500',
}

def load_predictions(file_path):
    """Return dict {image_name: (pred, true)} plus list of failed-parse images."""
    with open(file_path, encoding='utf-8-sig') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'results' in data:
        out, failed = {}, []
        for item in data['results']:
            m = re.search(r'Grade (\d)', item['predicted_output'])
            pred = int(m.group(1)) if m else -1
            if pred == -1:
                failed.append(item['image_name'])
            true_lab = item.get('true_grade', item.get('true_label'))
            out[item['image_name']] = (pred, true_lab)
        return out, failed
    out = {}
    for item in data:
        out[item['image_name']] = (item['predicted_label'], item['true_label'])
    return out, []

def mcnemar(dictA, dictB):
    """Image-keyed McNemar. Returns dict of stats; b+c==0 -> undefined (NaN p)."""
    common = set(dictA.keys()) & set(dictB.keys())
    a = b = c = d = 0
    t_mismatch = 0
    for img in common:
        pa, ta = dictA[img]
        pb, tb = dictB[img]
        if ta != tb:
            t_mismatch += 1
            continue
        ca, cb = (pa == ta), (pb == tb)
        if ca and cb:
            a += 1
        elif ca and not cb:
            b += 1
        elif not ca and cb:
            c += 1
        else:
            d += 1
    n = a + b + c + d
    acc_a = (a + b) / n * 100 if n else np.nan
    acc_b = (a + c) / n * 100 if n else np.nan
    if b + c == 0:
        return dict(a=a, b=b, c=c, d=d, n=n, acc_a=acc_a, acc_b=acc_b,
                    chi2_unc=np.nan, p_unc=np.nan, chi2_corr=np.nan, p_corr=np.nan,
                    t_mismatch=t_mismatch)
    chi2_unc = (b - c) ** 2 / (b + c)
    p_unc = chi2.sf(chi2_unc, df=1)
    chi2_corr = (abs(b - c) - 1) ** 2 / (b + c)
    p_corr = chi2.sf(chi2_corr, df=1)
    return dict(a=a, b=b, c=c, d=d, n=n, acc_a=acc_a, acc_b=acc_b,
                chi2_unc=chi2_unc, p_unc=p_unc, chi2_corr=chi2_corr, p_corr=p_corr,
                t_mismatch=t_mismatch)

def holm(pvals):
    """Holm-Bonferroni step-down; NaN (undefined tests) stay NaN."""
    p = np.array(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    defined = ~np.isnan(p)
    m = defined.sum()
    if m == 0:
        return out
    idx = np.where(defined)[0]
    order = idx[np.argsort(p[idx], kind='stable')]
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(prev, p[i] * (m - rank)))
        out[i] = adj
        prev = adj
    return out

def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return 'N/A'
    if p < 0.001:
        return 'P<.001'
    return 'P=%.3f' % p

def run_family(tag, entries):
    """entries: list of (label, full_path). Returns (df, preds, failed_report)."""
    preds, failed_report = {}, {}
    for label, path in entries:
        d, failed = load_predictions(path)
        preds[label] = d
        failed_report[label] = failed
    labels = list(preds.keys())
    rows = []
    for i, j in combinations(range(len(labels)), 2):
        A, B = labels[i], labels[j]
        st = mcnemar(preds[A], preds[B])
        rows.append(dict(Model_A=A, Model_B=B, **st))
    df = pd.DataFrame(rows)
    df['P_holm_unc'] = holm(df['p_unc'].values)
    df['P_holm_corr'] = holm(df['p_corr'].values)
    df['P_unc_fmt'] = df['p_unc'].apply(fmt_p)
    df['P_corr_fmt'] = df['p_corr'].apply(fmt_p)
    df['P_holm_unc_fmt'] = df['P_holm_unc'].apply(fmt_p)
    df['P_holm_corr_fmt'] = df['P_holm_corr'].apply(fmt_p)
    df['Sig_unc'] = df['p_unc'] < 0.05
    df['Sig_corr'] = df['p_corr'] < 0.05
    df['Sig_holm_unc'] = df['P_holm_unc'] < 0.05
    df['Sig_holm_corr'] = df['P_holm_corr'] < 0.05
    return df, preds, failed_report

def merge(orig, new, colA='Model_A', colB='Model_B'):
    m = orig.merge(new, left_on=[colA, colB], right_on=['Model_A', 'Model_B'],
                   how='outer', suffixes=('_orig', '_new'))
    return m

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_dir", default=os.path.join("..", "4-evalresult"),
                    help="directory of the 4-evalresult model output JSONs (Family 55)")
    ap.add_argument("--clipoa_json", default=os.path.join("..", "4-evalresult", "001-CLIP-OA.json"),
                    help="CLIP-OA evaluation JSON (Family 3)")
    ap.add_argument("--focal_json",
                    default=os.path.join("..", "8-clip-loss", "evaluation_results_clip_focal.json"),
                    help="focal-loss CLIP evaluation JSON (Family 3)")
    ap.add_argument("--weighted_json",
                    default=os.path.join("..", "8-clip-loss", "evaluation_results_detailed_weighted.json"),
                    help="weighted-loss CLIP evaluation JSON (Family 3)")
    ap.add_argument("--output_dir", default=os.path.join(".", "mcnemar_results"),
                    help="output directory for CSV files")
    ap.add_argument("--orig55_csv", default=None,
                    help="optional: original 55-comparison CSV (row-position pairing) for comparison")
    ap.add_argument("--orig3_csv", default=None,
                    help="optional: original 3-comparison CSV for comparison")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- Family 55 (Supplementary Table 3) ----------
    entries55 = [(NAMES55[f], os.path.join(args.eval_dir, f)) for f in MODELS55]

    print('=' * 90)
    print('STEP 1/3  Family 55 (Supplementary Table 3)')
    print('=' * 90)
    df55, preds55, fail55 = run_family('55', entries55)
    for lbl, failed in fail55.items():
        if failed:
            print('  parse failed (%d): %s' % (len(failed), lbl))
    sizes = {lbl: len(d) for lbl, d in preds55.items()}
    print('  image counts:', sizes)
    print('  total pairs:', len(df55))
    n_na = df55['p_unc'].isna().sum()
    print('  undefined (b+c=0) pairs:', n_na)
    print('  t_mismatch occurrences:', (df55['t_mismatch'] > 0).sum())

    print()
    print('=' * 90)
    print('STEP 2/3  Family 3 (Supplementary Table 4: CLIP-OA vs Focal/Weighted)')
    print('=' * 90)
    entries3 = [
        ('CLIP-OA', args.clipoa_json),
        ('Focal Loss', args.focal_json),
        ('Weighted Loss', args.weighted_json),
    ]
    df3, preds3, fail3 = run_family('3', entries3)
    print('  image counts:', {lbl: len(d) for lbl, d in preds3.items()})
    print('  total pairs:', len(df3))

    print()
    print('=' * 90)
    print('STEP 3/3  Merge with original results & comparison (optional)')
    print('=' * 90)

    cols_keep = ['Model_A', 'Model_B', 'n', 'a', 'b', 'c', 'd', 'acc_a', 'acc_b',
                 'chi2_unc', 'p_unc', 'P_unc_fmt', 'chi2_corr', 'p_corr', 'P_corr_fmt',
                 'P_holm_unc', 'P_holm_unc_fmt', 'P_holm_corr', 'P_holm_corr_fmt',
                 'Sig_unc', 'Sig_corr', 'Sig_holm_unc', 'Sig_holm_corr']
    df55[cols_keep].round(6).to_csv(os.path.join(args.output_dir, 'mcnemar_55_image_paired.csv'),
                                    index=False)
    df3[cols_keep].round(6).to_csv(os.path.join(args.output_dir, 'mcnemar_3_image_paired.csv'),
                                   index=False)

    if args.orig55_csv:
        orig55 = pd.read_csv(args.orig55_csv)
        m55 = merge(orig55, df55)
        m55['Orig_Sig'] = m55['p-value'] < 0.05
        m55['Sig_changed_unc'] = (m55['Orig_Sig'] != m55['Sig_unc'])
        m55['Sig_changed_holm_unc'] = (m55['Orig_Sig'] != m55['Sig_holm_unc'])
        print('  [55-family] significance changed vs original (raw uncorr):',
              int(m55['Sig_changed_unc'].sum()), 'of', len(m55))
        print('  [55-family] significance changed vs original (holm uncorr):',
              int(m55['Sig_changed_holm_unc'].sum()), 'of', len(m55))
        m55.round(6).to_csv(os.path.join(args.output_dir, 'comparison_55_orig_vs_new.csv'),
                            index=False)
    else:
        print('  --orig55_csv not provided; skipping original 55-comparison merge')

    if args.orig3_csv:
        orig3 = pd.read_csv(args.orig3_csv)
        orig3 = orig3.rename(columns={'Model A': 'Model_A', 'Model B': 'Model_B'})
        m3 = merge(orig3, df3)
        m3['Orig_Sig'] = m3['p-value'] < 0.05
        m3['Sig_changed_corr'] = (m3['Orig_Sig'] != m3['Sig_corr'])
        m3['Sig_changed_holm_corr'] = (m3['Orig_Sig'] != m3['Sig_holm_corr'])
        print('  [3-family] significance changed vs original (corr):',
              int(m3['Sig_changed_corr'].sum()), 'of', len(m3))
        print('  [3-family] significance changed vs original (holm corr):',
              int(m3['Sig_changed_holm_corr'].sum()), 'of', len(m3))
        m3.round(6).to_csv(os.path.join(args.output_dir, 'comparison_3_orig_vs_new.csv'),
                           index=False)
    else:
        print('  --orig3_csv not provided; skipping original 3-comparison merge')

    print()
    print('Outputs written to', os.path.abspath(args.output_dir))
    print('  - mcnemar_55_image_paired.csv')
    print('  - mcnemar_3_image_paired.csv')
    print('  - comparison_55_orig_vs_new.csv (only if --orig55_csv provided)')
    print('  - comparison_3_orig_vs_new.csv (only if --orig3_csv provided)')
    print('DONE')

if __name__ == "__main__":
    main()
