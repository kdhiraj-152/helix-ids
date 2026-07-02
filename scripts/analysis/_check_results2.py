import pandas as pd, numpy as np

# Architecture results
print("=== Cross-Architecture Validation ===")
df = pd.read_csv("results/phase53/architecture_generalization.csv")
print(df[["architecture","mean_off_diag_mf1","n_params"]].to_string(index=False))
print(f"Range: {df['mean_off_diag_mf1'].max():.4f} - {df['mean_off_diag_mf1'].min():.4f} = {df['mean_off_diag_mf1'].max()-df['mean_off_diag_mf1'].min():.4f}")
print(f"All architectures positive transfer: {(df['mean_off_diag_mf1']>0.3).sum()}/{len(df)}")

print()

# Feature results
print("=== Feature Set Generalization ===")
df2 = pd.read_csv("results/phase53/feature_generalization.csv")
print(df2[["feature_config","n_features","mean_off_diag_mf1"]].to_string(index=False))
baseline = df2[df2["feature_config"]=="canonical_17"]["mean_off_diag_mf1"].values[0]
print(f"Baseline (17 feats): {baseline:.4f}")
print(f"Reduced 10: {df2[df2['feature_config']=='reduced_10']['mean_off_diag_mf1'].values[0]:.4f} (Δ={df2[df2['feature_config']=='reduced_10']['mean_off_diag_mf1'].values[0]-baseline:+.4f})")
print(f"Min features {df2['n_features'].min()} -> MF1={df2.loc[df2['n_features'].idxmin(), 'mean_off_diag_mf1']:.4f}")
