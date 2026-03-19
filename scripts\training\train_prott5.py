import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# =========================================================
# ?숋툘 [?ㅼ젙] ProtT5??寃쎈줈 ?섏젙
# =========================================================
EMBEDDING_DIR = './data/embeddings/protT5_xl'

# ?뚯씪紐?(data/embeddings/protT5_xl ?대? ?뚯씪紐??뺤씤 ?꾩슂, ?놁쑝硫??섏젙?댁빞 ??
TRAIN_EMB_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_emb.npy')
TRAIN_IDS_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_ids.npy')
TEST_EMB_FILE  = os.path.join(EMBEDDING_DIR, 'testsuperset_emb.npy')
TEST_IDS_FILE  = os.path.join(EMBEDDING_DIR, 'testsuperset_ids.npy')

TRAIN_TERMS = './data/raw/train/train_terms.tsv'
OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ProtT5-XL 李⑥썝 (湲곕낯 1024)
INPUT_DIM = 1024 
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 10
NUM_WORKERS = 0

SAVE_THRESHOLD = 0.01  
TOP_K = 100            

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"?뵦 Device: {device} | Loading ProtT5 Embeddings...")

# =========================================================
# ?룛截??곗씠??濡쒕뜑 (AggregatedDataset ?ъ궗??
# =========================================================
class AggregatedDataset(Dataset):
    def __init__(self, emb_matrix, id_map, target_pids, labels_dict=None, num_classes=0):
        self.emb_matrix = emb_matrix  
        self.id_map = id_map          
        self.target_pids = target_pids 
        self.labels_dict = labels_dict
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.target_pids)
    
    def __getitem__(self, idx):
        pid = self.target_pids[idx]
        if pid in self.id_map:
            row_idx = self.id_map[pid]
            emb = self.emb_matrix[row_idx]
        else:
            emb = np.zeros(INPUT_DIM, dtype=np.float32)
            
        emb = torch.tensor(emb, dtype=torch.float32)

        if self.labels_dict is not None:
            label = np.zeros(self.num_classes, dtype=np.float32)
            if pid in self.labels_dict:
                for t_idx in self.labels_dict[pid]: label[t_idx] = 1.0
            return emb, torch.tensor(label)
        else:
            return emb, pid

# =========================================================
# ?쭬 紐⑤뜽 & Loss
# =========================================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x): return self.layers(x)

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip = gamma_neg, gamma_pos, clip
    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_neg = (1 - x_sigmoid + self.clip).clamp(max=1)
        loss = -1 * (self.gamma_pos * y * torch.log(x_sigmoid.clamp(min=1e-8)) + 
                     self.gamma_neg * (1 - y) * torch.log(xs_neg.clamp(min=1e-8)) * (xs_neg**self.gamma_neg))
        return loss.sum()

def clean_ids(id_array):
    id_list = []
    flat_array = id_array.reshape(-1)
    for pid in flat_array:
        if isinstance(pid, bytes):
            pid = pid.decode('utf-8')
        pid = str(pid).strip().replace('>', '')
        if '|' in pid:
            parts = pid.split('|')
            if len(parts) >= 2:
                pid = parts[1]
        id_list.append(pid)
    return id_list

print("?봽 ?곗씠??濡쒕뵫 諛?ID 留ㅼ묶 ?쒖옉...")

# Train ?곗씠??濡쒕뱶
print(f"   Shape Check Target: {TRAIN_EMB_FILE}")
train_emb = np.load(TRAIN_EMB_FILE, mmap_mode='r')
if train_emb.shape[1] != INPUT_DIM:
    print(f"?좑툘 Warning: Dimension Mismatch! Code expects {INPUT_DIM}, but file is {train_emb.shape}")
    INPUT_DIM = train_emb.shape[1] # ?먮룞 蹂댁젙

train_ids_raw = np.load(TRAIN_IDS_FILE)
train_ids = clean_ids(train_ids_raw)
train_id_map = {pid: i for i, pid in enumerate(train_ids)}

# Test ?곗씠??濡쒕뱶
test_emb = np.load(TEST_EMB_FILE, mmap_mode='r')
test_ids_raw = np.load(TEST_IDS_FILE)
test_ids = clean_ids(test_ids_raw)
test_id_map = {pid: i for i, pid in enumerate(test_ids)}

# Term 以鍮?
train_terms = pd.read_csv(TRAIN_TERMS, sep='\t')
term_counts = train_terms['term'].value_counts()
top_terms = term_counts[term_counts >= 10].index.tolist()
print(f"   ?렞 Target Terms: {len(top_terms)}")

term2idx = {t: i for i, t in enumerate(top_terms)}
idx2term = {i: t for i, t in enumerate(top_terms)}

# Label 留듯븨
train_labels = {}
for pid, term in zip(train_terms['EntryID'], train_terms['term']):
    pid = str(pid).strip()
    if term in term2idx and pid in train_id_map: 
        if pid not in train_labels: train_labels[pid] = []
        train_labels[pid].append(term2idx[term])

# Split
valid_pids = list(train_labels.keys())
trn_ids, val_ids = train_test_split(valid_pids, test_size=0.1, random_state=42)

def train_and_predict(name, criterion, output_file):
    print(f"\n?? [{name}] Training Start...")
    train_ds = AggregatedDataset(train_emb, train_id_map, trn_ids, train_labels, len(top_terms))
    val_ds   = AggregatedDataset(train_emb, train_id_map, val_ids, train_labels, len(top_terms))
    test_ds  = AggregatedDataset(test_emb,  test_id_map,  test_ids, num_classes=0)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    model = SimpleMLP(INPUT_DIM, len(top_terms)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            
    print(f"?뵰 [{name}] Predicting...")
    model.eval()
    with open(output_file, 'w') as f:
        with torch.no_grad():
            for x, pids in tqdm(test_loader, desc="Inference"):
                x = x.to(device)
                probs = torch.sigmoid(model(x)).cpu().numpy()
                for i, pid in enumerate(pids):
                    p = probs[i]
                    indices = np.where(p >= SAVE_THRESHOLD)[0]
                    scores = p[indices]
                    if len(scores) > TOP_K:
                        top_idx = np.argsort(scores)[-TOP_K:]
                        indices = indices[top_idx]
                        scores = scores[top_idx]
                    for idx, score in zip(indices, scores):
                        f.write(f"{pid}\t{idx2term[idx]}\t{score:.5f}\n")
    print(f"??Saved: {output_file}")
    return output_file

# =========================================================
# ?뵦 ?ㅽ뻾
# =========================================================
bce_out = f"{OUTPUT_DIR}/pred_bce_prott5.tsv"
if not os.path.exists(bce_out):
    train_and_predict("BCE_Head", nn.BCEWithLogitsLoss(), bce_out)

asl_out = f"{OUTPUT_DIR}/pred_asl_prott5.tsv"
if not os.path.exists(asl_out):
    train_and_predict("ASL_Tail", AsymmetricLoss(), asl_out)

# Rank Ensemble
print("\n?뵕 Rank Ensemble (ProtT5)...")
df_bce = pd.read_csv(bce_out, sep='\t', names=['Id', 'Term', 'Score'], header=None)
df_asl = pd.read_csv(asl_out, sep='\t', names=['Id', 'Term', 'Score'], header=None)
df_bce['Rank'] = df_bce['Score'].rank(pct=True)
df_asl['Rank'] = df_asl['Score'].rank(pct=True)

merged = pd.merge(df_bce, df_asl, on=['Id', 'Term'], suffixes=('_bce', '_asl'), how='outer').fillna(0)
merged['Score'] = (merged['Rank_bce'] + merged['Rank_asl']) / 2

final_out = f"{OUTPUT_DIR}/submission_prott5_step1.tsv"
merged[['Id', 'Term', 'Score']].to_csv(final_out, sep='\t', index=False, header=False)

print(f"?럦 ProtT5 ?꾨즺: {final_out}")

