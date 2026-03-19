import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm

# =========================================================
# ?숋툘 [?ㅼ젙] 寃쎈줈 ?뺤씤
# =========================================================
CLEAN_DL_FILE = './results/submission_Kingdom_Filtered.tsv'
CLEAN_DIA_FILE = './results/submission_diamond_taxon_filtered.tsv'
GAF_POS_FILE = './gaf_positive_preds.tsv'
GAF_NEG_FILE = './gaf_negative_preds.tsv'
TEST_IDS_FILE = './data/embeddings/testsuperset_ids.npy'
OUTPUT_FILE = './results/submission_SOTA_Kingdom_Max_LowRAM.tsv'
TEMP_FILE = './results/temp_merged.tsv' # 以묎컙 ??μ슜

# =========================================================

def clean_id_str(pid):
    if isinstance(pid, bytes): pid = pid.decode('utf-8')
    pid = str(pid).strip().replace('>', '')
    if '|' in pid: 
        parts = pid.split('|')
        pid = parts[1] if len(parts) >= 2 else pid
    return pid

print("?? Final Merge: Kingdom DL + Clean Diamond (Low RAM Mode)...")

# 1. Test ID 由ъ뒪??濡쒕뱶 (寃利앹슜 - ?닿굔 硫붾え由ъ뿉 ?щ젮????
print(f"   ?뱥 Loading Valid Test IDs...")
if not os.path.exists(TEST_IDS_FILE):
    print(f"?슚 Test ID ?뚯씪 ?놁쓬: {TEST_IDS_FILE}")
    exit()

raw_ids = np.load(TEST_IDS_FILE)
# Set? 硫붾え由щ? 醫 ?곗?留?寃???띾룄媛 O(1)?대씪 ?꾩닔
valid_test_ids = set([clean_id_str(pid) for pid in raw_ids])
print(f"     -> {len(valid_test_ids):,} valid IDs loaded.")

# 硫붾え由??뺣낫
del raw_ids
gc.collect()

# 2. ?ㅽ듃由?蹂묓빀 (Stream Merge)
# ?뺤뀛?덈━?????ｌ? ?딄퀬, ?쇰떒 ?뚯씪濡????잛븘遺볦뒿?덈떎.
# ?섏쨷??sort濡?以묐났???쒓굅?섎뒗 諛⑹떇???⑥뵮 硫붾え由щ? ?곴쾶 ?곷땲??

print("   ??Streaming Merge (Writing to disk directly)...")

with open(TEMP_FILE, 'w') as f_out:
    # (1) DL ?뚯씪 ?쎌쑝硫댁꽌 ?곌린
    if os.path.exists(CLEAN_DL_FILE):
        print(f"     Processing DL: {CLEAN_DL_FILE}")
        with open(CLEAN_DL_FILE, 'r') as f_in:
            for line in tqdm(f_in):
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                pid = clean_id_str(parts[0])
                if pid not in valid_test_ids: continue
                # 洹몃?濡?? (?섏쨷??Max 泥섎━)
                f_out.write(f"{pid}\t{parts[1]}\t{parts[2]}\n")
    
    # (2) Diamond ?뚯씪 ?쎌쑝硫댁꽌 ?곌린
    if os.path.exists(CLEAN_DIA_FILE):
        print(f"     Processing Diamond: {CLEAN_DIA_FILE}")
        with open(CLEAN_DIA_FILE, 'r') as f_in:
            for line in tqdm(f_in):
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                pid = clean_id_str(parts[0])
                if pid not in valid_test_ids: continue
                f_out.write(f"{pid}\t{parts[1]}\t{parts[2]}\n")

    # (3) GAF Positive (1.0) ?쎌쑝硫댁꽌 ?곌린
    if os.path.exists(GAF_POS_FILE):
        print(f"     Processing GAF Positive: {GAF_POS_FILE}")
        with open(GAF_POS_FILE, 'r') as f_in:
            for line in f_in:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                pid = clean_id_str(parts[0])
                if pid in valid_test_ids:
                    f_out.write(f"{pid}\t{parts[1]}\t1.00000\n")

print("   ?뮶 Sorting & Aggregating (This is the heavy part, but safe)...")

# 3. Pandas Chunk 泥섎━ (硫붾え由???컻 諛⑹?)
# ?뚯씪???⑹뼱由?Chunk)濡??쎌뼱??Max Aggregation ?섑뻾
chunk_size = 5_000_000  # 500留?以꾩뵫 泥섎━ (硫붾え由??곹솴???곕씪 議곗젅)
temp_results = {}

for chunk in tqdm(pd.read_csv(TEMP_FILE, sep='\t', header=None, names=['Id', 'Term', 'Score'], chunksize=chunk_size)):
    # Chunk ?댁뿉??Max ?섑뻾
    # (媛숈? ID-Term???щ윭 以꾩뿉 ?⑹뼱???덉뼱??寃곌뎅???⑹퀜吏묐땲??
    grouped = chunk.groupby(['Id', 'Term'])['Score'].max()
    
    for (pid, term), score in grouped.items():
        if (pid, term) in temp_results:
            # 湲곗〈 媛믨낵 鍮꾧탳?댁꽌 ?щ㈃ ?낅뜲?댄듃
            if score > temp_results[(pid, term)]:
                temp_results[(pid, term)] = score
        else:
            if score > 0.01: # 理쒖냼 ?먯닔 而?                temp_results[(pid, term)] = score

# ?꾩떆 ?뚯씪 ??젣
if os.path.exists(TEMP_FILE):
    os.remove(TEMP_FILE)

# 4. GAF Negative ?곸슜 (硫붾え由ъ뿉??諛붾줈 ??젣)
if os.path.exists(GAF_NEG_FILE):
    print("   ?㏏ Applying GAF Negative Filter...")
    with open(GAF_NEG_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            pid = clean_id_str(parts[0])
            term = parts[1]
            # ?뺤뀛?덈━?먯꽌 諛붾줈 ??젣 (留ㅼ슦 鍮좊쫫)
            if (pid, term) in temp_results:
                del temp_results[(pid, term)]

# 5. 理쒖쥌 ???print(f"   ?뮶 Final Saving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    for (pid, term), score in tqdm(temp_results.items()):
        f.write(f"{pid}\t{term}\t{score:.5f}\n")

print(f"?럦 Low-RAM Merge ?꾨즺! ?쒖텧 ?뚯씪: {OUTPUT_FILE}")
