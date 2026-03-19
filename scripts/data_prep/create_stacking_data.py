import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Config
TRAIN_FASTA = "./data/raw/train/train_sequences.fasta"
TRAIN_TERMS = "./data/raw/train/train_terms.tsv"
TRAIN_TAXONOMY = "./data/raw/train/train_taxonomy.tsv"
DIAMOND_DB = "train_diamond.dmnd"
DIAMOND_OUT = "train_self_hits.tsv"

def build_diamond_db():
    print("?뵪 Building Diamond DB from Train Sequences...")
    # Using diamond makedb
    # Assuming 'diamond' is in path or we use wsl diamond
    cmd = f"wsl diamond makedb --in {TRAIN_FASTA} -d {DIAMOND_DB}"
    os.system(cmd)

def run_diamond_self():
    print("?룂 Running Diamond (Train vs Train)...")
    # Output format: qseqid sseqid pident length mismatch ...
    # We need: qseqid sseqid pident evalue bitscore
    # Exclude self-hits later or here? Diamond reports self hits.
    # We will filter qseqid == sseqid in python.
    cmd = f"wsl diamond blastp -q {TRAIN_FASTA} -d {DIAMOND_DB} -o {DIAMOND_OUT} --outfmt 6 qseqid sseqid pident evalue bitscore --sensitive -k 2" # k=2 to get at least 1 non-self match
    os.system(cmd)

def process_data():
    print("?뱤 Processing Hits & Creating Features...")
    
    # Load Labels
    print("   Loading Truth...")
    # Truth: ID -> Set of GO Terms
    truth = {}
    df_terms = pd.read_csv(TRAIN_TERMS, sep="\t", names=["id", "term", "aspect"])
    # Group by ID
    groups = df_terms.groupby("id")
    for pid, group in tqdm(groups, desc="Grouping Terms"):
        truth[pid] = set(group["term"].values)
        
    # Load Hits
    print("   Loading Diamond Hits...")
    # columns: qseqid, sseqid, pident, evalue, bitscore
    # Stream it
    
    features = []
    
    with open(DIAMOND_OUT, "r") as f:
        for line in tqdm(f, desc="Parsing Hits"):
            parts = line.strip().split('\t')
            if len(parts) < 5: continue
            
            q_id = parts[0]
            s_id = parts[1]
            
            if q_id == s_id: continue # Skip self
            
            pident = float(parts[2])
            evalue = float(parts[3])
            bitscore = float(parts[4])
            
            # Label Logic:
            # Does the Subject (s_id) carry ANY valid annotation for Query (q_id)?
            # Wait, Stacking usually predicts "Is THIS specific term correct?".
            # But Diamond gives "Protein Match".
            # We are learning: "Trust this Neighbor?"
            # Label = Jaccard Similarity derived? Or simply "Is pident trusted?"
            # Let's define Label: Precision of transfer.
            # If we transfer ALL terms from s_id to q_id, what is the F1?
            # Or simpler: Is s_id a "useful" homolog? (Jaccard > 0.3)
            
            # Actually, standard CAFA homolog transfer: Score = pident * constant.
            # We want to learn: Score = Model(pident, evalue, bitscore).
            # Target = Jaccard(Truth[q_id], Truth[s_id])
            
            if q_id not in truth or s_id not in truth: continue
            
            t_q = truth[q_id]
            t_s = truth[s_id]
            
            intersection = len(t_q.intersection(t_s))
            union = len(t_q.union(t_s))
            jaccard = intersection / union if union > 0 else 0.0
            
            # Feature Vector
            features.append([pident, np.log1p(evalue), bitscore, jaccard])
            
    # Save as CSV
    print(f"?뮶 Saving Stacking Data ({len(features)} rows)...")
    df_feat = pd.DataFrame(features, columns=["pident", "log_evalue", "bitscore", "target_jaccard"])
    df_feat.to_csv("stacking_train_data.csv", index=False)
    print("??Done.")

if __name__ == "__main__":
    build_diamond_db()
    run_diamond_self()
    process_data()

