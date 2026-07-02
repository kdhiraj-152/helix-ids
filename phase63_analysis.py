#!/usr/bin/env python3
"""Phase 63 comprehensive analysis report."""
import pandas as pd, json, numpy as np

base = '/Users/kdhiraj/Downloads/RP-2/results/phase63'
plasticity = pd.read_csv(f'{base}/plasticity_results.csv')
forgetting = pd.read_csv(f'{base}/forgetting_results.csv')
transfer = pd.read_csv(f'{base}/transfer_results.csv')
drift = pd.read_csv(f'{base}/representation_drift.csv')
layer = pd.read_csv(f'{base}/layerwise_updates.csv')

# Map codes to labels
method_names = plasticity[['condition','method']].drop_duplicates().set_index('condition')['method'].to_dict()
cond_list = ['A', 'B', 'C', 'D', 'E', 'F']

def hdr(title):
    print(f"\n{'='*70}")
    print(title)
    print('='*70)

# 1. All-dataset avg MF1 by stage
hdr("1. ALL-DATASET AVG MACRO F1 BY CONDITION x STAGE")
avg_mf1 = plasticity.groupby(['condition', 'stage'])['macro_f1'].mean().round(4)
print(f"{'Cond':5s} {'Method':45s} {'S0':>6s} {'S1':>6s} {'S2':>6s} {'S3':>6s}")
print('-'*75)
for c in cond_list:
    name = method_names[c]
    vals = [avg_mf1.xs((c, f'stage_{i}')) for i in range(4)]
    print(f"  {c:3s} {name:45s} {vals[0]:.4f} {vals[1]:.4f} {vals[2]:.4f} {vals[3]:.4f}")

# 2. Per-dataset Stage 3
hdr("2. STAGE 3 PER-DATASET MACRO F1")
s3 = plasticity[plasticity['stage'] == 'stage_3']
ds_order = ['nsl_kdd','unsw_nb15','cicids2017','cicids2018','ton_iot','bot_iot','iot23','kyoto2006','ugr16']
print(f"{'Cond':5s} {'NSL-KDD':>8s} {'UNSW-NB15':>9s} {'CIC2017':>8s} {'CIC2018':>8s} {'TON-IoT':>8s} {'Bot-IoT':>8s} {'IoT-23':>8s} {'Kyoto+':>8s} {'UGR16':>8s} {'Avg':>8s}")
print('-'*85)
for c in cond_list:
    vals = []
    for ds in ds_order:
        row = s3[(s3['condition']==c) & (s3['dataset']==ds)]
        vals.append(row['macro_f1'].values[0] if len(row)>0 else np.nan)
    avg = np.nanmean(vals)
    line = f"  {c:3s}"
    for v in vals:
        line += f" {v:8.4f}" if not np.isnan(v) else f" {'N/A':>8s}"
    line += f" {avg:.4f}"
    print(line)

# 3. Forgetting
hdr("3. FORGETTING (avg over all datasets)")
print(f"{'Cond':5s} {'Method':45s} {'Avg Forget':>10s} {'Max Forget':>10s}")
print('-'*72)
for _, row in forgetting.iterrows():
    c = row['condition']
    nm = method_names.get(c, c)
    print(f"  {c:3s} {nm:45s} {row['mean_forgetting']:10.4f} {row['max_forgetting']:10.4f}")

# 4. BWT
hdr("4. BACKWARD TRANSFER (Stage 3 BWT + Stability)")
trans_s3 = transfer[(transfer['metric']=='avg_bwt') & (transfer['dataset']=='stage_3')]
print(f"{'Cond':5s} {'Method':45s} {'BWT':>10s} {'Stability':>10s}")
print('-'*72)
for _, row in trans_s3.iterrows():
    c = row['condition']
    nm = method_names.get(c, c)
    stab = transfer[(transfer['condition']==c) & (transfer['metric']=='stability') & (transfer['dataset']=='stage_3')]
    sval = stab['value'].values[0] if len(stab)>0 else float('nan')
    print(f"  {c:3s} {nm:45s} {row['value']:10.4f} {sval:10.4f}")

# 5. Representation drift (CKA)
hdr("5. REPRESENTATION DRIFT (Avg CKA)")
if 'cka' in drift.columns:
    drift_avg = drift.groupby('condition')['cka'].mean()
    s3_cka = drift[drift['stage']=='stage_3'].groupby('condition')['cka'].mean()
    print(f"{'Cond':5s} {'Method':45s} {'All CKA':>8s} {'S3 CKA':>8s}")
    print('-'*68)
    for c in cond_list:
        nm = method_names.get(c, c)
        print(f"  {c:3s} {nm:45s} {drift_avg.get(c,float('nan')):.4f} {s3_cka.get(c,float('nan')):.4f}")

# 6. Layerwise
hdr("6. LAYERWISE COSINE SIMILARITY (Stage 3)")
ls3 = layer[layer['stage']=='stage_3']
if len(ls3)>0:
    print(f"{'Cond':5s} {'Method':45s} {'CosSim':>8s} {'UpdMag':>10s}")
    print('-'*70)
    for c in cond_list:
        nm = method_names.get(c, c)
        cos = ls3[ls3['condition']==c]['cosine_similarity'].mean()
        upd = ls3[ls3['condition']==c]['update_magnitude'].mean()
        print(f"  {c:3s} {nm:45s} {cos:.4f} {upd:.6f}")

# 7. Computational
hdr("7. COMPUTATIONAL COST")
if 'total_seconds' in plasticity.columns:
    cost = plasticity[['condition','total_seconds','lora_trainable','lora_total']].drop_duplicates().dropna(subset=['total_seconds'])
    print(f"{'Cond':5s} {'Method':45s} {'Time(s)':>8s} {'LoRA Tr':>10s} {'LoRA Tot':>10s}")
    print('-'*80)
    for _, row in cost.iterrows():
        c = row['condition']
        nm = method_names.get(c, c)
        print(f"  {c:3s} {nm:45s} {row['total_seconds']:8.0f} {int(row['lora_trainable']):>10,} {int(row['lora_total']):>10,}")

# 8. Key findings
hdr("8. KEY SCIENTIFIC FINDINGS")
baseline_s3 = avg_mf1.xs(('A', 'stage_3'))
baseline_s2 = avg_mf1.xs(('A', 'stage_2'))

print(f"\nBaseline (A, frozen backbone) Stage 3 avg MF1: {baseline_s3:.4f}")
print(f"Baseline (A, frozen backbone) Stage 2 avg MF1: {baseline_s2:.4f}")
print()

best_s3 = -1; best_name = ''
for c in cond_list:
    v = avg_mf1.xs((c, 'stage_3'))
    d = v - baseline_s3
    marker = ' << BEST' if v > best_s3 else ''
    if v > best_s3: best_s3, best_name = v, c
    print(f"  {c}: Stage 3 MF1={v:.4f} (Δ={d:+.4f}){marker}")

print(f"\nSTAGE 2 (Kyoto2006+ adaptation) - WHERE PLASTICITY HELPED:")
for c in ['B','C','D','E','F']:
    v = avg_mf1.xs((c, 'stage_2'))
    d_pct = ((v - baseline_s2) / baseline_s2) * 100
    print(f"  {c} ({method_names[c]}): Stage 2 MF1={v:.4f} (Δ={d_pct:+.1f}% vs A)")

print(f"\nSTAGE 3 (UGR16 adaptation) - COLLAPSE:")
for c in cond_list:
    v = avg_mf1.xs((c, 'stage_3'))
    d_s2 = avg_mf1.xs((c, 'stage_2')) - v
    print(f"  {c}: Stage 3 MF1={v:.4f} (drop from Stage 2: {d_s2:.4f})")

# Summary
hdr("9. SUMMARY")
print("""
  FINDING 1: Plasticity helps WITHIN-stage
    - Stage 2 (Kyoto2006+): D (full LoRA) MF1=0.3925 (+29.7% vs A)
      F (replay) MF1=0.4577 (+51.3% vs A)
    - Confirms Phase 52 outcome: representation adaptation improves 
      within-dataset performance

  FINDING 2: Plasticity does NOT prevent cross-stage forgetting
    - Stage 3 (UGR16): ALL conditions converge to ~0.30 MF1
    - A, B, E, F: identical 0.3043
    - C, D: worse at 0.2822 (full LoRA overfits the previous dataset)
    - The benefit from plasticity at Stage 2 is completely lost at Stage 3

  FINDING 3: Full-weight unfreezing (B, C) = frozen (A) at Stage 3
    - Despite making 66K-165K backbone parameters trainable, Stage 3
      MF1 is identical to frozen baseline
    - The bottleneck is representation incompatibility (P(Y|X) mismatch),
      not parameter overwriting from frozen weights

  FINDING 4: Replay prevents forgetting at Stage 1->2 transition
    - F preserves IoT-23 knowledge through Stage 2 (MF1=0.9540 vs 0.2892)
    - But Stage 3 (UGR16) still catastrophically erases everything
    - 1824 exemplars / 242 KB insufficient for large distribution shift

  FINDING 5: P(Y|X) bottleneck confirmed
    - The conditional distribution learned from NSL-KDD is fundamentally
      incompatible with the sequence of external distributions
    - Adaptation-only strategies cannot overcome cumulative representation
      divergence in sequential CL for NIDS
    - Pre-training on multi-dataset joint embedding space may be needed
""")
