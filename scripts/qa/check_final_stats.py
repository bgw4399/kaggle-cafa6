import pandas as pd
import numpy as np
import sys
import os

FILE_TARGET = sys.argv[1] if len(sys.argv) > 1 else "./results/final_submission/submission_Final_Repaired.tsv"

def main():
    print(f"📊 Final Verification for: {FILE_TARGET}")
    if not os.path.exists(FILE_TARGET):
        print("❌ File not found.")
        return

    chunk_size = 1000000
    
    total_rows = 0
    unique_pids = set()
    min_score = 1.0
    max_score = 0.0
    
    # Histogram bins: 0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
    hist = np.zeros(5, dtype=int)
    
    invalid_score_count = 0
    
    print("   🚀 Reading chunks...")
    
    for df in pd.read_csv(FILE_TARGET, sep='\t', header=None, names=['id', 'term', 'score'], chunksize=chunk_size):
        # Update Counts
        total_rows += len(df)
        unique_pids.update(df['id'].unique())
        
        # Stats
        s = df['score'].values
        min_score = min(min_score, s.min())
        max_score = max(max_score, s.max())
        
        # Invalid check
        invalid = (s < 0.0) | (s > 1.0)
        invalid_score_count += np.sum(invalid)
        
        # Histogram
        # 0.0 <= s < 0.2 -> idx 0
        # ...
        # 0.8 <= s <= 1.0 -> idx 4
        # Be careful with edges
        indices = np.floor(s * 5).astype(int)
        indices = np.clip(indices, 0, 4)
        
        # Fast histogram for this chunk
        chunk_hist = np.bincount(indices, minlength=5)
        hist += chunk_hist
        
    print("\n✅ Verification Complete.")
    print(f"   Total Predictions: {total_rows:,}")
    print(f"   Unique Proteins: {len(unique_pids):,}")
    print(f"   Avg Preds/Prot: {total_rows / len(unique_pids):.1f}")
    print(f"   Score Range: [{min_score:.5f}, {max_score:.5f}]")
    print(f"   Invalid Scores: {invalid_score_count}")
    
    print("\n   📈 Score Distribution:")
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    for i, count in enumerate(hist):
        pct = count / total_rows * 100 if total_rows > 0 else 0
        print(f"     {labels[i]}: {count:,} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
