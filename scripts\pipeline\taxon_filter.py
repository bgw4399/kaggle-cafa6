import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# [?ㅼ젙] V3 BCE 紐⑤뜽 ?ъ슜
INPUT_PRED_FILE = './results/pred_esm2_15B_v3_BCE.tsv' 
OUTPUT_FILE = './results/submission_V3_Taxon_Filtered.tsv'

# ?뵦 蹂듦뎄 ?꾧퀎媛?(???먯닔 ?댁긽?대㈃ ?숈뒿 ?곗씠?곗뿉 ?놁뼱???대┝)
RESTORE_THRESHOLD = 0.75

TRAIN_TAXON_IDX = './data/embeddings/taxonomy/train_species_idx.npy'
TRAIN_TERMS = './data/raw/train/train_terms.tsv'
TRAIN_ID_FILE = './data/embeddings/esm2_15B/train_sequences_ids.npy'
TEST_IDS = './data/embeddings/testsuperset_ids.npy'
TEST_TAXON_IDX = './data/embeddings/taxonomy/test_species_idx.npy'

def clean_id_str(pid):
    if isinstance(pid, bytes): pid = pid.decode('utf-8')
    pid = str(pid).strip().replace('>', '')
    if '|' in pid: parts = pid.split('|'); pid = parts[1] if len(parts) >= 2 else pid
    return pid

print("?? Soft Taxonomy Filtering Start...")

# 1. 洹쒖튃 濡쒕뱶
raw_train_ids = np.load(TRAIN_ID_FILE)
train_ids = [clean_id_str(x) for x in raw_train_ids]
train_species = np.load(TRAIN_TAXON_IDX)
id2sp = dict(zip(train_ids, train_species))

sp2terms = {}
train_df = pd.read_csv(TRAIN_TERMS, sep='\t')
for pid, term in tqdm(zip(train_df['EntryID'], train_df['term']), total=len(train_df)):
    pid = str(pid).strip()
    if pid in id2sp:
        s = id2sp[pid]
        if s not in sp2terms: sp2terms[s] = set()
        sp2terms[s].add(term)

# 2. Test ID ?뺣낫
raw_test_ids = np.load(TEST_IDS)
test_pids = [clean_id_str(x) for x in raw_test_ids]
test_species = np.load(TEST_TAXON_IDX)
test_id2sp = dict(zip(test_pids, test_species))

# 3. Soft Filtering
with open(INPUT_PRED_FILE, 'r') as f_in, open(OUTPUT_FILE, 'w') as f_out:
    for line in tqdm(f_in):
        parts = line.strip().split('\t')
        if len(parts) < 3: continue
        
        pid = clean_id_str(parts[0])
        term = parts[1]
        try: score = float(parts[2])
        except: continue
        
        is_valid = True
        if pid in test_id2sp:
            sp = test_id2sp[pid]
            if sp in sp2terms:
                if term not in sp2terms[sp]:
                    # ?놁?留??먯닔媛 ?믪쑝硫??대┝!
                    if score >= RESTORE_THRESHOLD:
                        is_valid = True 
                    else:
                        is_valid = False
        else:
            continue # Test ID ?꾨땲硫?Skip
        
        if is_valid:
            f_out.write(f"{pid}\t{term}\t{score:.5f}\n")

print(f"?뮶 Soft Filtered DL: {OUTPUT_FILE}")

