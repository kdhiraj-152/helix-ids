import pandas as pd, numpy as np
df = pd.read_csv("results/phase53/seed_stability.csv")
print("=== Seed Stability Results ===")
print(df[["seed","mean_off_diag_mf1","n_params"]].to_string(index=False))
print(f"\nMean MF1: {df['mean_off_diag_mf1'].mean():.4f}")
print(f"Std MF1: {df['mean_off_diag_mf1'].std():.4f}")
print(f"Variance: {df['mean_off_diag_mf1'].var():.6f}")
print(f"95% CI: [{df['mean_off_diag_mf1'].quantile(0.025):.4f}, {df['mean_off_diag_mf1'].quantile(0.975):.4f}]")
print(f"Min: {df['mean_off_diag_mf1'].min():.4f}, Max: {df['mean_off_diag_mf1'].max():.4f}")
print(f"Catastrophic failures (MF1<0.2): {(df['mean_off_diag_mf1']<0.2).sum()}")
print(f"All seeds MF1>0: {(df['mean_off_diag_mf1']>0).sum()}/{len(df)}")
# CV
print(f"CV (std/mean): {df['mean_off_diag_mf1'].std()/df['mean_off_diag_mf1'].mean():.4f}")
