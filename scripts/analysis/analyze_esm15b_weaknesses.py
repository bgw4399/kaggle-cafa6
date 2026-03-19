import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os

# Files
STEP1_FILE = "./results/submission_step1_aggregated.tsv"
BEST_FILE = "./results/submission_378.tsv"
OBO_FILE = "./data/raw/train/go-basic.obo"

def load_go_namespaces(obo_path):
    """Load GO term namespaces (BP, MF, CC)"""
    namespaces = {}
    names = {}
    term = None
    with open(obo_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '[Term]':
                term = None
            elif line.startswith('id: '):
                term = line[4:].split()[0]
            elif line.startswith('namespace: ') and term:
                ns = line[11:]
                if 'biological_process' in ns:
                    namespaces[term] = 'BP'
                elif 'molecular_function' in ns:
                    namespaces[term] = 'MF'
                elif 'cellular_component' in ns:
                    namespaces[term] = 'CC'
            elif line.startswith('name: ') and term:
                names[term] = line[6:]
    return namespaces, names

def analyze_step1():
    print("?뵇 Analyzing ESM15B Model Predictions (submission_step1_aggregated.tsv)...")
    
    # Load GO namespaces
    print("\n?뱴 Loading GO ontology...")
    namespaces, names = load_go_namespaces(OBO_FILE)
    print(f"   -> Loaded {len(namespaces)} GO terms")
    
    # Load Step1 file (sample)
    print(f"\n?뱰 Loading Step1 predictions (sample)...")
    step1_df = pd.read_csv(STEP1_FILE, sep='\t', header=None, 
                           names=['id', 'term', 'score'], nrows=10_000_000)
    print(f"   -> Loaded {len(step1_df):,} rows")
    
    # Load Best file for comparison (sample)
    print(f"\n?뱰 Loading Best_378 for comparison...")
    best_df = pd.read_csv(BEST_FILE, sep='\t', header=None, 
                          names=['id', 'term', 'score'], nrows=10_000_000)
    print(f"   -> Loaded {len(best_df):,} rows")
    
    # Analysis 1: Namespace distribution
    print("\n" + "="*60)
    print("?뱤 1. GO Namespace Distribution")
    print("="*60)
    
    step1_df['namespace'] = step1_df['term'].map(namespaces)
    best_df['namespace'] = best_df['term'].map(namespaces)
    
    step1_ns = step1_df['namespace'].value_counts(normalize=True)
    best_ns = best_df['namespace'].value_counts(normalize=True)
    
    print("\n   Step1 (ESM15B):")
    for ns, pct in step1_ns.items():
        print(f"     {ns}: {pct*100:.1f}%")
    
    print("\n   Best_378:")
    for ns, pct in best_ns.items():
        print(f"     {ns}: {pct*100:.1f}%")
    
    # Analysis 2: Score distribution by namespace
    print("\n" + "="*60)
    print("?뱤 2. Score Distribution by Namespace")
    print("="*60)
    
    for ns in ['BP', 'MF', 'CC']:
        step1_ns_scores = step1_df[step1_df['namespace'] == ns]['score']
        best_ns_scores = best_df[best_df['namespace'] == ns]['score']
        
        print(f"\n   {ns}:")
        print(f"     Step1: Mean={step1_ns_scores.mean():.3f}, >0.5: {(step1_ns_scores > 0.5).mean()*100:.1f}%")
        print(f"     Best:  Mean={best_ns_scores.mean():.3f}, >0.5: {(best_ns_scores > 0.5).mean()*100:.1f}%")
    
    # Analysis 3: Prediction count per protein
    print("\n" + "="*60)
    print("?뱤 3. Predictions Per Protein by Namespace")
    print("="*60)
    
    for ns in ['BP', 'MF', 'CC']:
        step1_per_prot = step1_df[step1_df['namespace'] == ns].groupby('id').size()
        best_per_prot = best_df[best_df['namespace'] == ns].groupby('id').size()
        
        print(f"\n   {ns}:")
        print(f"     Step1: Mean={step1_per_prot.mean():.1f}")
        print(f"     Best:  Mean={best_per_prot.mean():.1f}")
    
    # Analysis 4: Top GO terms with low confidence in Step1 but high in Best
    print("\n" + "="*60)
    print("?뱤 4. Terms with Major Score Differences")
    print("="*60)
    
    # Aggregate by term
    step1_term_scores = step1_df.groupby('term')['score'].agg(['mean', 'count'])
    best_term_scores = best_df.groupby('term')['score'].agg(['mean', 'count'])
    
    # Find terms where Best has much higher scores
    common_terms = set(step1_term_scores.index) & set(best_term_scores.index)
    
    diffs = []
    for term in common_terms:
        step1_mean = step1_term_scores.loc[term, 'mean']
        best_mean = best_term_scores.loc[term, 'mean']
        diff = best_mean - step1_mean
        if best_term_scores.loc[term, 'count'] > 100:  # Common term
            diffs.append((term, step1_mean, best_mean, diff, namespaces.get(term, '?'), names.get(term, '?')))
    
    diffs.sort(key=lambda x: x[3], reverse=True)
    
    print("\n   Terms where Best_378 >> Step1 (ESM15B is weak):")
    for term, s1, b, d, ns, name in diffs[:15]:
        print(f"     {term} ({ns}): Step1={s1:.2f} vs Best={b:.2f} (?={d:+.2f})")
        print(f"        -> {name[:50]}...")
    
    # Analysis 5: High frequency terms with low scores
    print("\n" + "="*60)
    print("?뱤 5. Common Terms with Low Confidence in Step1")
    print("="*60)
    
    # Common terms = high count, low mean score
    step1_weak = step1_term_scores[step1_term_scores['count'] > 500].sort_values('mean').head(20)
    
    print("\n   High-frequency terms with lowest mean scores in Step1:")
    for term in step1_weak.index:
        mean = step1_weak.loc[term, 'mean']
        count = step1_weak.loc[term, 'count']
        ns = namespaces.get(term, '?')
        name = names.get(term, '?')[:40]
        print(f"     {term} ({ns}): Mean={mean:.3f}, Count={count:,}")
        print(f"        -> {name}...")

if __name__ == "__main__":
    analyze_step1()


