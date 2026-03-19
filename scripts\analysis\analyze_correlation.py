import pandas as pd
import numpy as np

# Files
ESM_FILE = "./results/submission_378_Optimized.tsv"
PROT_FILE = "./results/pred_prott5_resmlp_focal.tsv"
ANKH_FILE = "./results/pred_ankh_resmlp_focal.tsv"

def load_sample(path, n=500000):
    print(f"?뱰 Loading {path}...")
    try:
        df = pd.read_csv(path, sep='\t', header=None, names=['id', 'term', 'score'], nrows=n)
        return df
    except:
        return None

print("?? Analyzing Correlation for Ensemble...")

df_esm = load_sample(ESM_FILE)
df_prot = load_sample(PROT_FILE)
df_ankh = load_sample(ANKH_FILE)

# Merge on ID+Term
print("   Merging outcomes...")
m1 = pd.merge(df_esm, df_prot, on=['id', 'term'], suffixes=('_esm', '_prot'), how='inner')
final = pd.merge(m1, df_ankh, on=['id', 'term'], how='inner')
final.rename(columns={'score': 'score_ankh'}, inplace=True)

print(f"   Common Predictions: {len(final):,}")

# Correlation
corr = final[['score_esm', 'score_prot', 'score_ankh']].corr()
print("\n?뱤 Pearson Correlation Matrix:")
print(corr)

if len(final) < 1000:
    print("?좑툘 Warning: Too few common predictions. Models might be predicting very different terms!")

