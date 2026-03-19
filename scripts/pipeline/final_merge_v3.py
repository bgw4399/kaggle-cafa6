import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm

# =========================================================
# ?숋툘 [?ㅼ젙] V3 Pipeline - GAF ?놁씠 ?덉쟾?섍쾶
# =========================================================
CLEAN_DL_FILE = './results/submission_V3_Taxon_Filtered.tsv'
CLEAN_DIA_FILE = './results/submission_diamond_taxon_filtered.tsv'
TEST_IDS_FILE = './data/embeddings/testsuperset_ids.npy'
OUTPUT_FILE = './results/submission_V3_Final.tsv'
TEMP_FILE = './results/temp_merged.tsv'

# GAF ?뚯씪 ?놁씠 吏꾪뻾 (洹쒖튃 以??
USE_GAF = False

def clean_id_str(pid):
    if isinstance(pid, bytes): pid = pid.decode('utf-8')
    pid = str(pid).strip().replace('>', '')
    if '|' in pid: 
        parts = pid.split('|')
        pid = parts[1] if len(parts) >= 2 else pid
    return pid

print("?? Final Merge: V3 DL + Diamond (No GAF)...")

# 1. Test ID 濡쒕뱶
print(f"   ?뱥 Loading Valid Test IDs...")
raw_ids = np.load(TEST_IDS_FILE)
valid_test_ids = set([clean_id_str(pid) for pid in raw_ids])
print(f"     -> {len(valid_test_ids):,} valid IDs loaded.")
del raw_ids
gc.collect()

# 2. ?ㅽ듃由?蹂묓빀
print("   ??Streaming Merge...")
with open(TEMP_FILE, 'w') as f_out:
    # DL ?뚯씪
    if os.path.exists(CLEAN_DL_FILE):
        print(f"     Processing DL: {CLEAN_DL_FILE}")
        with open(CLEAN_DL_FILE, 'r') as f_in:
            for line in tqdm(f_in):
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                pid = clean_id_str(parts[0])
                if pid not in valid_test_ids: continue
                f_out.write(f"{pid}\t{parts[1]}\t{parts[2]}\n")
    
    # Diamond ?뚯씪
    if os.path.exists(CLEAN_DIA_FILE):
        print(f"     Processing Diamond: {CLEAN_DIA_FILE}")
        with open(CLEAN_DIA_FILE, 'r') as f_in:
            for line in tqdm(f_in):
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                pid = clean_id_str(parts[0])
                if pid not in valid_test_ids: continue
                f_out.write(f"{pid}\t{parts[1]}\t{parts[2]}\n")

# 3. Aggregation (Max)
print("   ?뮶 Aggregating (Max)...")
chunk_size = 5_000_000
temp_results = {}

for chunk in tqdm(pd.read_csv(TEMP_FILE, sep='\t', header=None, names=['Id', 'Term', 'Score'], chunksize=chunk_size)):
    grouped = chunk.groupby(['Id', 'Term'])['Score'].max()
    for (pid, term), score in grouped.items():
        if (pid, term) in temp_results:
            if score > temp_results[(pid, term)]:
                temp_results[(pid, term)] = score
        else:
            if score > 0.01:
                temp_results[(pid, term)] = score

if os.path.exists(TEMP_FILE):
    os.remove(TEMP_FILE)

# 4. ???
print(f"   ?뮶 Saving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    for (pid, term), score in tqdm(temp_results.items()):
        f.write(f"{pid}\t{term}\t{score:.5f}\n")

print(f"?럦 V3 Final Complete: {OUTPUT_FILE}")


