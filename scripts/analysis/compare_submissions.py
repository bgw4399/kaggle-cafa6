import pandas as pd
import numpy as np
import sys
from collections import defaultdict

# Files to compare
FILES = {
    "Best_378": "./results/submission_378.tsv",
    "Step1_Old": "./results/submission_step1_aggregated.tsv",
    "V3_New": "./results/pred_esm2_15B_v3_BCE.tsv",
}

def analyze_file(name, path):
    print(f"\n{'='*60}")
    print(f"?뱤 Analyzing: {name}")
    print(f"   Path: {path}")
    print(f"{'='*60}")
    
    try:
        # Load sample (first 10M lines)
        df = pd.read_csv(path, sep='\t', header=None, names=['id', 'term', 'score'], nrows=10_000_000)
    except FileNotFoundError:
        print(f"   ??File not found")
        return
    except Exception as e:
        print(f"   ??Error: {e}")
        return

    total_rows = len(df)
    unique_proteins = df['id'].nunique()
    unique_terms = df['term'].nunique()
    
    # Predictions per protein
    preds_per_protein = df.groupby('id').size()
    
    print(f"\n?뱢 Basic Stats:")
    print(f"   Total Rows (sample): {total_rows:,}")
    print(f"   Unique Proteins: {unique_proteins:,}")
    print(f"   Unique Terms: {unique_terms:,}")
    
    print(f"\n?뱤 Predictions Per Protein:")
    print(f"   Mean: {preds_per_protein.mean():.1f}")
    print(f"   Median: {preds_per_protein.median():.1f}")
    print(f"   Max: {preds_per_protein.max()}")
    print(f"   Min: {preds_per_protein.min()}")
    
    print(f"\n?뱤 Score Distribution:")
    print(f"   Mean: {df['score'].mean():.4f}")
    print(f"   Median: {df['score'].median():.4f}")
    print(f"   >0.9: {(df['score'] > 0.9).sum():,} ({100*(df['score'] > 0.9).sum()/len(df):.1f}%)")
    print(f"   >0.5: {(df['score'] > 0.5).sum():,} ({100*(df['score'] > 0.5).sum()/len(df):.1f}%)")
    print(f"   <0.1: {(df['score'] < 0.1).sum():,} ({100*(df['score'] < 0.1).sum()/len(df):.1f}%)")

def main():
    for name, path in FILES.items():
        analyze_file(name, path)
    
    print("\n" + "="*60)
    print("?렞 Summary: If 'Repaired' has much higher Predictions Per Protein")
    print("   than 'Best_378', that's likely causing precision drop.")
    print("="*60)

if __name__ == "__main__":
    main()

