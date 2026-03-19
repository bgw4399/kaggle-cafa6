import pandas as pd
import numpy as np
import sys

# === ?ㅼ젙 (寃쎈줈留??섏젙?섏꽭?? ===
# 1. 0.378???뚯씪 (Old Best)
FILE_OLD = "./results/submission_378.tsv"
WEIGHT_OLD = 0.7

# 2. 0.32???뚯씪 (New Scientific Filtered)
FILE_NEW = "./results/final_submission/submission_final_filtered.tsv"
WEIGHT_NEW = 0.3

# 3. 寃곌낵 ?뚯씪
OUTPUT_FILE = "./results/final_submission/submission_Master_Ensemble.tsv"

def main():
    print("?? Master Ensemble (High-RAM Vectorized) Starting...")
    print(f"   Old Best (w={WEIGHT_OLD}): {FILE_OLD}")
    print(f"   New Best (w={WEIGHT_NEW}): {FILE_NEW}")

    # 1. Load Files (Pandas Fast Read)
    print("?뱿 Loading Old File (0.378)...")
    # Header handling: Assuming NO header. If header exists, change header=None to header=0
    df_old = pd.read_csv(FILE_OLD, sep="\t", names=["id", "term", "score"], header=None, dtype={"score": float})
    
    print("?뱿 Loading New File (0.32)...")
    df_new = pd.read_csv(FILE_NEW, sep="\t", names=["id", "term", "score"], header=None, dtype={"score": float})
    
    print(f"   Old: {len(df_old):,} rows | New: {len(df_new):,} rows")

    # 2. Merge (Outer Join on ID, Term)
    print("?봽 Merging DataFrames (This requires 16GB+ RAM)...")
    # Outer merge preserves all predictions from both sides
    merged = pd.merge(df_old, df_new, on=["id", "term"], how="outer", suffixes=('_old', '_new'))
    
    # Fill NaN with 0.0
    merged["score_old"] = merged["score_old"].fillna(0.0)
    merged["score_new"] = merged["score_new"].fillna(0.0)

    # 3. Calculate Weighted Score
    print("?㎜ Calculating Weighted Scores...")
    # Vectorized calculation
    merged["final_score"] = (merged["score_old"] * WEIGHT_OLD) + (merged["score_new"] * WEIGHT_NEW)

    # 4. Filter & Sort
    print("?㏏ Filtering Low Scores (< 0.001)...")
    final_df = merged[merged["final_score"] > 0.001]
    
    print("   Final Count:", len(final_df))
    
    # 5. Save
    print(f"?뮶 Saving to {OUTPUT_FILE}...")
    final_df[["id", "term", "final_score"]].to_csv(
        OUTPUT_FILE, 
        sep="\t", 
        index=False, 
        header=False, 
        float_format="%.5f"
    )
    
    print("??Done! Master Ensemble Created.")

if __name__ == "__main__":
    main()

