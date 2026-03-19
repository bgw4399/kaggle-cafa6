import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# Files
PRED_ESM = "./results/scientific/esm2_15b/val_pred_esm.tsv"
PRED_PROT = "./results/scientific/prott5/val_pred_prott5.tsv"
GT_FILE = "./data/raw/train/Split/val_terms_split.tsv"
OBO_FILE = "./data/raw/train/go-basic.obo"

# Config
NAMESPACE = 'BPO' # Optimize for BPO as it's the main challenge
ROOT_TERM = 'GO:0008150'

def load_obo(path):
    print("?뱴 Loading OBO...")
    parents = defaultdict(set)
    obj = {}
    with open(path, 'r') as f:
        term = None
        ns = None
        for line in f:
            line = line.strip()
            if line.startswith('id: GO:'):
                term = line[4:].split()[0]
            elif line.startswith('namespace: '):
                ns = line[11:]
                if ns == 'biological_process': ns = 'BPO'
                elif ns == 'molecular_function': ns = 'MFO'
                elif ns == 'cellular_component': ns = 'CCO'
                obj[term] = ns
            elif line.startswith('is_a: ') and term:
                p = line[6:].split(' ! ')[0]
                parents[term].add(p)
    
    ancestors = defaultdict(set)
    def get_ancestors(t):
        if t in ancestors: return ancestors[t]
        ans = set()
        for p in parents[t]:
            ans.add(p)
            ans.update(get_ancestors(p))
        ancestors[t] = ans
        return ans

    sys.setrecursionlimit(50000)
    for t in list(obj.keys()):
        get_ancestors(t)
    return ancestors, obj

def load_preds(path):
    print(f"   ?뱿 Loading {path}...")
    preds = defaultdict(dict) # pid -> {term: score}
    with open(path, 'r') as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p)<3: continue
            preds[p[0]][p[1]] = float(p[2])
    return preds

def load_gt(path, ancestors, target_ns):
    print("   ?뱿 Loading Ground Truth...")
    gt = defaultdict(set)
    df = pd.read_csv(path, sep='\t')
    for pid, term in zip(df['EntryID'], df['term']):
        # Propagate
        gt[pid].add(term)
        if term in ancestors:
            gt[pid].update(ancestors[term])
            
    # Filter valid proteins (Must have at least 1 annotation in target_ns)
    valid_gt = {}
    for pid, terms in gt.items():
        # Check if any term is in target_ns (naive check or using obj_map)
        # We need obj_map.
        # But efficiently: we check later.
        valid_gt[pid] = terms
    return valid_gt

def calc_fmax(gt, preds, obj_map, ns):
    valid_pids = []
    ns_gt = {}
    
    # Filter GT for NS
    for pid, terms in gt.items():
        st = {t for t in terms if obj_map.get(t) == ns}
        if st:
            valid_pids.append(pid)
            ns_gt[pid] = st
            
    thresholds = np.arange(0.01, 1.01, 0.02)
    best_f = 0.0
    
    # Pre-calculate pred sets for this NS
    ns_preds = {}
    for pid in valid_pids:
        if pid in preds:
            ns_preds[pid] = {t: s for t, s in preds[pid].items() if obj_map.get(t) == ns}
        else:
            ns_preds[pid] = {}

    for t in thresholds:
        sum_prec = 0.0
        sum_rec = 0.0
        count = 0
        
        for pid in valid_pids:
            g = ns_gt[pid]
            p_map = ns_preds.get(pid, {})
            p_set = {term for term, score in p_map.items() if score >= t}
            
            n_g = len(g)
            n_p = len(p_set)
            n_inter = len(g.intersection(p_set))
            
            prec = n_inter / n_p if n_p > 0 else 0.0
            rec = n_inter / n_g if n_g > 0 else 0.0
            
            sum_prec += prec
            sum_rec += rec
            count += 1
            
        avg_prec = sum_prec / count
        avg_rec = sum_rec / count
        
        if avg_prec + avg_rec > 0:
            f = (2 * avg_prec * avg_rec) / (avg_prec + avg_rec)
            if f > best_f: best_f = f
            
    return best_f

def main():
    ancestors, obj_map = load_obo(OBO_FILE)
    gt_map = load_gt(GT_FILE, ancestors, NAMESPACE)
    
    p1 = load_preds(PRED_ESM)
    p2 = load_preds(PRED_PROT)
    
    # Common PIDs (Union)
    all_pids = set(p1.keys()).union(set(p2.keys()))
    
    # Weights to try
    weights = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0] # ESM Weight
    
    print("\n?뽳툘 Optimization Start (Target: BPO)...")
    
    results = []
    
    for w in tqdm(weights, desc="Testing Weights"):
        # Mix Predictions
        # Linear Combination: Score = w*ESM + (1-w)*Prot
        mixed = defaultdict(dict)
        
        # Iterate all pids
        # To affect F-max, we process valid_gt pids mostly, but let's do all
        # Optimization: Only compute for proteins in GT? Yes.
        # But we need to load them first. `calc_fmax` does filtering.
        # To be safe, mix all keys.
        
        # Optimized mixing:
        # Iterate p1, p2
        # Use Dict union logic
        
        # Since this loop is inside, let's just do it for valid_gt pids
        # Filter GT again?
        
        # Just mix strictly
        keys = set(p1.keys()).union(p2.keys())
        for pid in keys:
            terms = set(p1.get(pid, {}).keys()).union(p2.get(pid, {}).keys())
            for term in terms:
                s1 = p1.get(pid, {}).get(term, 0.0)
                s2 = p2.get(pid, {}).get(term, 0.0)
                score = (s1 * w) + (s2 * (1-w))
                mixed[pid][term] = score
                
        # Propagate Max
        # This is expensive inside loop.
        # Strategy: Propagate INDIVIDUALLY first?
        # Linear combination of propagated scores?
        #   Prop(A+B) != Prop(A) + Prop(B) because Prop is MAX.
        #   max(A+B) vs max(A)+max(B).
        #   Actually, normally we average the *Raw* scores, then propagate?
        #   Or Average the *Propagated* scores?
        #   Consensus in CAFA: Usually average raw, then propagate.
        #   But here `p1` and `p2` (from load_preds) are likely RAW (leaf) predictions from MLP.
        #   So we mix raw, then propagate.
        
        final_preds = defaultdict(dict)
        for pid, p_map in mixed.items():
            canvas = p_map.copy()
            for term, score in p_map.items():
                if term in ancestors:
                    for anc in ancestors[term]:
                        if score > canvas.get(anc, 0.0):
                            canvas[anc] = score
            final_preds[pid] = canvas
            
        fmax = calc_fmax(gt_map, final_preds, obj_map, NAMESPACE)
        print(f"   ESM={w:.1f}, Prot={1-w:.1f} -> F-max: {fmax:.4f}")
        results.append((w, fmax))
        
    best_w, best_f = max(results, key=lambda x: x[1])
    print(f"\n?룇 Best Configuration: ESM={best_w:.1f}, Prot={1-best_w:.1f} (F-max={best_f:.4f})")

if __name__ == "__main__":
    main()

