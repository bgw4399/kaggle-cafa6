import pandas as pd
import numpy as np
import argparse
import os
import sys
import gc
import shutil

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def split_file(path, prefix, n_buckets):
    print(f"🔪 Splitting {path} into {n_buckets} buckets...", flush=True)
    handles = [open(f"{prefix}_{i}.tsv", "w") for i in range(n_buckets)]
    
    count = 0
    with open(path, "r") as f:
        for line in f:
            parts = line.split("\t", 1)
            if not parts[0]: continue
            
            # Simple hash
            pid = parts[0]
            # remove > if present
            clean_pid = pid.replace(">", "")
            
            h = hash(clean_pid) % n_buckets
            handles[h].write(line)
            count += 1
            if count % 1000000 == 0:
                print(f"   Processed {count:,} lines...", flush=True, end="\r")
    
    for h in handles:
        h.close()
    print(f"\n   Done. Total {count:,} lines.")

def merge_bucket(i, base_path, text_path, out_handle, args):
    print(f"🔄 Processing Bucket {i}...", flush=True)
    
    # Check bounds
    # Load Base
    if os.path.exists(base_path):
        df_base = pd.read_csv(base_path, sep='\t', header=None, names=['id', 'term', 'score'], 
                              dtype={'score': 'float32', 'id': 'str', 'term': 'str'},
                              usecols=[0, 1, 2])
    else:
        df_base = pd.DataFrame(columns=['id', 'term', 'score'])

    # Load Text
    if os.path.exists(text_path):
        df_text = pd.read_csv(text_path, sep='\t', header=None, names=['id', 'term', 'score'], 
                              dtype={'score': 'float32', 'id': 'str', 'term': 'str'},
                              usecols=[0, 1, 2])
    else:
        df_text = pd.DataFrame(columns=['id', 'term', 'score'])
        
    if len(df_base) == 0 and len(df_text) == 0:
        return

    # Calculate Gates (from Base)
    if len(df_base) > 0:
        strength = df_base.groupby('id')['score'].max()
        gates = sigmoid(args.gate_k * (strength - args.gate_mid))
    else:
        gates = pd.Series(dtype='float32')

    # Merge
    merged = pd.merge(df_base, df_text, on=['id', 'term'], how='outer', suffixes=('_base', '_text'))
    
    merged['score_base'] = merged['score_base'].fillna(0.0)
    merged['score_text'] = merged['score_text'].fillna(0.0)
    
    # Map gates
    default_gate = 1 / (1 + np.exp(-(args.gate_k * (0.0 - args.gate_mid))))
    
    # Create final score
    merged['gg'] = merged['id'].map(gates).fillna(default_gate).astype('float32')
    
    # Safety: Protect Base (SOTA) -> max(base, text*weight)
    merged['final_score'] = np.maximum(
        merged['score_base'],
        merged['score_text'] * (1.0 - merged['gg'])
    )
    
    # Write
    merged = merged[merged['final_score'] >= 0.001]
    
    for row in merged[['id', 'term', 'final_score']].itertuples(index=False):
        out_handle.write(f"{row.id}\t{row.term}\t{row.final_score:.5f}\n")
    
    del df_base, df_text, merged, gates
    gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--prott5", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate_k", type=float, default=10.0)
    parser.add_argument("--gate_mid", type=float, default=0.70)
    parser.add_argument("--buckets", type=int, default=10)
    parser.add_argument("--temp_dir", default="temp_merge_safe")
    args = parser.parse_args()

    os.makedirs(args.temp_dir, exist_ok=True)
    
    # 1. Split
    split_file(args.baseline, os.path.join(args.temp_dir, "base"), args.buckets)
    split_file(args.prott5, os.path.join(args.temp_dir, "text"), args.buckets)
    
    # 2. Merge
    print(f"📝 Merging to {args.out}...")
    with open(args.out, "w") as out_f:
        for i in range(args.buckets):
            base_p = os.path.join(args.temp_dir, f"base_{i}.tsv")
            text_p = os.path.join(args.temp_dir, f"text_{i}.tsv")
            merge_bucket(i, base_p, text_p, out_f, args)
    
    # 3. Cleanup
    print("🧹 Cleanup...")
    shutil.rmtree(args.temp_dir)
    print("🎉 All Done. Safe Merge Complete.")

if __name__ == "__main__":
    main()
