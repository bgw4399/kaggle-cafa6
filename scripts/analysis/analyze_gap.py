import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import matplotlib.pyplot as plt

# ?ㅼ젙
BEST_FILE = "./results/submission_378.tsv"
TRAIN_TERMS = "./data/raw/train/train_terms.tsv"
OBO_FILE = "./data/raw/train/go-basic.obo"

def load_obo(path):
    print("?뱴 Loading Ontology...")
    term_ns = {}
    with open(path, 'r', encoding='utf-8') as f:
        term = None
        for line in f:
            line = line.strip()
            if line.startswith('id: '): term = line[4:].split()[0]
            elif line.startswith('namespace: ') and term:
                term_ns[term] = line[11:]
    return term_ns

def analyze_gap():
    print(f"?? Analyze Gap to 0.46 (Current: 0.378)...")
    
    # 1. Load Submission
    print("   ?뱰 Loading submission...")
    df = pd.read_csv(BEST_FILE, sep='\t', header=None, names=['id', 'term', 'score'], nrows=20_000_000)
    
    # 2. Score Histograms
    print("   ?뱤 Calculating Score Distribution...")
    scores = df['score'].values
    
    print(f"     Mean Score: {np.mean(scores):.4f}")
    print(f"     Median Score: {np.median(scores):.4f}")
    print(f"     std dev: {np.std(scores):.4f}")
    
    # 3. Term Frequency Analysis
    print("   ?뱤 Analyzing Term Frequencies...")
    term_counts = df['term'].value_counts()
    print(f"     Total Unique Predicted Terms: {len(term_counts):,}")
    print(f"     Top 10 Frequent Terms:")
    print(term_counts.head(10))
    
    # 4. Namespace Coverage
    term_ns = load_obo(OBO_FILE)
    df['ns'] = df['term'].map(term_ns)
    ns_counts = df['ns'].value_counts(normalize=True)
    print("\n   ?뱤 Namespace Coverage:")
    print(ns_counts)
    
    # 5. Information Content (Proxy) - Rare vs Frequent
    # Assuming rare terms count < 1000 in predictions
    rare_terms = term_counts[term_counts < 1000].index
    frequent_terms = term_counts[term_counts >= 1000].index
    
    rare_score = df[df['term'].isin(rare_terms)]['score'].mean()
    freq_score = df[df['term'].isin(frequent_terms)]['score'].mean()
    
    print(f"\n   ?뱤 Rare vs Frequent Analysis:")
    print(f"     Frequent Terms (>1000 preds): {len(frequent_terms):,} terms, Mean Score = {freq_score:.4f}")
    print(f"     Rare Terms (<1000 preds): {len(rare_terms):,} terms, Mean Score = {rare_score:.4f}")
    
    print("\n   ?쭬 [Insight]")
    if rare_score < 0.2 and freq_score > 0.4:
        print("     -> Model is biased towards Frequent Terms. F-max penalty for missing specific/rare terms.")
    if len(term_counts) < 10000:
        print("     -> Prediction vocabulary is too small. Top rankers predict 20k-30k terms.")

if __name__ == "__main__":
    analyze_gap()


