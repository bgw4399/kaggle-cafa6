import pandas as pd
import numpy as np

FILE_SOTA = "./results/submission_SOTA_Ensemble_Final.tsv"
FILE_BEST = "./results/submission_378.tsv"

def analyze(path, name):
    print(f"?뱤 Analyzing {name}...")
    # Read first 5M rows for speed estimation
    df = pd.read_csv(path, sep='\t', header=None, names=['id', 'term', 'score'], nrows=5_000_000)
    
    scores = df['score'].values
    unique_terms = df['term'].nunique()
    unique_prots = df['id'].nunique()
    preds_per_prot = len(df) / unique_prots
    
    print(f"   Score Mean:   {np.mean(scores):.4f}")
    print(f"   Score Median: {np.median(scores):.4f}")
    print(f"   Score Std:    {np.std(scores):.4f}")
    print(f"   Preds/Prot:   {preds_per_prot:.1f}")
    print(f"   Unique Terms: {unique_terms:,}")
    print(f"   > 0.5 Ratio:  {(scores > 0.5).mean()*100:.1f}%")
    return np.mean(scores), unique_terms

print("?? Comparing Best_378 vs SOTA_Ensemble...")
print("-" * 40)
m1, t1 = analyze(FILE_BEST, "Best_378 (Baseline)")
print("-" * 40)
m2, t2 = analyze(FILE_SOTA, "SOTA_Ensemble (Final)")
print("-" * 40)

print("?쭬 Insight Analysis:")
if t2 > t1:
    print(f"   ??Diversity Boost: Found {t2 - t1:,} NEW terms.")
else:
    print(f"   ?좑툘 Diversity Drop: Lost {t1 - t2:,} terms.")

if m2 < m1:
    print("   ??Calibration: Scores are more conservative (less False Positives).")
else:
    print("   ?좑툘 Aggression: Scores generally increased.")

