import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import multiprocessing

# Config
OBO_FILE = "./data/raw/train/go-basic.obo"
GT_FILE = "./data/raw/train/Split/val_terms_split.tsv"
PRED_FILE = "./results/scientific/prott5/val_pred_prott5.tsv"
ASPECTS = ['BPO', 'CCO', 'MFO']
ROOTS = {'BPO': 'GO:0008150', 'CCO': 'GO:0005575', 'MFO': 'GO:0003674'}

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
    
    # Propagate Ancestors
    print("   ?㎚ Building Ancestors Graph...")
    ancestors = defaultdict(set)
    def get_ancestors(t):
        if t in ancestors: return ancestors[t]
        ans = set()
        for p in parents[t]:
            ans.add(p)
            ans.update(get_ancestors(p))
        ancestors[t] = ans
        return ans

    # Iterate all terms to build cache
    # To avoid recursion limit, use iterative? CAFA terms ~40k. Recursion might hit limit.
    # Python default is 1000. 
    sys.setrecursionlimit(50000)
    for t in list(obj.keys()):
        get_ancestors(t)
        
    return ancestors, obj

def propagate_gt(df, ancestors, item_col='EntryID', term_col='term'):
    print("   ?뱦 Propagating Ground Truth...")
    gt = defaultdict(set)
    for pid, term in zip(df[item_col], df[term_col]):
        if term in ancestors:
            gt[pid].add(term)
            gt[pid].update(ancestors[term])
        else:
            gt[pid].add(term)
    return gt

def propagate_preds(pred_path, ancestors):
    print("   ?뵰 Loading & Propagating Predictions...")
    preds = defaultdict(dict)
    with open(pred_path, 'r') as f:
        for line in tqdm(f):
            p = line.strip().split('\t')
            if len(p)<3: continue
            pid, term, score = p[0], p[1], float(p[2])
            
            # MaxProp logic: Parent score = max(Parent, Child)
            # We first store raw scores
            if score > preds[pid].get(term, 0.0):
                preds[pid][term] = score
                
    # Now propagate max
    # Efficient Prop: Topological sort is best, but iterative pass is easier.
    # However, ancestors map goes Up. 
    # For each term, update all ancestors?
    # Yes: for term t with score s: for a in ancestors[t]: preds[pid][a] = max(preds[pid][a], s)
    
    # Optimize: Pre-calculate flat ancestors?
    # Or just iterate predictions.
    final_preds = defaultdict(dict)
    
    for pid, p_map in tqdm(preds.items(), desc="Propagating Scores"):
        # Copy original
        canvas = p_map.copy()
        
        # Propagate Up
        for term, score in p_map.items():
            if term in ancestors:
                for anc in ancestors[term]:
                    if score > canvas.get(anc, 0.0):
                        canvas[anc] = score
        final_preds[pid] = canvas
        
    return final_preds

def calc_fmax(gt, preds, obj_map):
    print("   ?뱤 Calculating F-max...")
    
    thresholds = np.arange(0.01, 1.01, 0.02) # 50 steps
    fmax_res = {}
    
    # Pre-filter by Aspect
    for ns in ASPECTS:
        print(f"     Target: {ns}")
        # Build GT subset
        # Filter GT and Preds to this namespace
        # Actually standard CAFA evaluation does F-max per target protein, then avg.
        # But we need to separate namespaces.
        
        # Valid Proteins for this namespace? (Proteins that HAVE at least 1 term in this NS in GT?)
        # CAFA rule: Evaluation set is proteins with >=1 annotated term in that namespace.
        
        valid_pids = []
        ns_gt = {}
        for pid, terms in gt.items():
            st = {t for t in terms if obj_map.get(t) == ns}
            if st:
                valid_pids.append(pid)
                ns_gt[pid] = st
                
        if not valid_pids:
            print(f"     Create: No valid GT proteins for {ns}")
            continue
            
        print(f"       Proteins: {len(valid_pids)}")
        
        best_f = 0.0
        best_t = 0.0
        
        for t in tqdm(thresholds, leave=False):
            sum_prec = 0.0
            sum_rec = 0.0
            count = 0
            
            for pid in valid_pids:
                # GT set
                g = ns_gt[pid]
                n_g = len(g)
                
                # Pred set
                p_map = preds.get(pid, {})
                # Filter by NS and Threshold
                p_set = {term for term, score in p_map.items() 
                         if score >= t and obj_map.get(term) == ns}
                
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
                if f > best_f:
                    best_f = f
                    best_t = t
                    
        print(f"       ?룇 F-max: {best_f:.4f} (Tau={best_t:.2f})")
        fmax_res[ns] = (best_f, best_t)
        
    return fmax_res

def main():
    if len(sys.argv) > 1:
        global PRED_FILE
        PRED_FILE = sys.argv[1]
        
    ancestors, obj_map = load_obo(OBO_FILE)
    
    df_gt = pd.read_csv(GT_FILE, sep='\t')
    gt = propagate_gt(df_gt, ancestors)
    
    preds = propagate_preds(PRED_FILE, ancestors)
    
    res = calc_fmax(gt, preds, obj_map)
    
    print("\n??Final Results:")
    for ns, (f, t) in res.items():
        print(f"  {ns}: {f:.4f} (Tau={t:.2f})")

if __name__ == "__main__":
    main()

