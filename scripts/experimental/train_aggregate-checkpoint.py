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
# ?숋툘 [?ㅼ젙] 紐⑤뜽蹂꾨줈 二쇱꽍 ??댁꽌 ?ъ슜?섏꽭??
# =========================================================
# 1. Ankh Large (異붿쿇!)
# MODEL_NAME = "ankh_large"
# EMBEDDING_DIR = './data/embeddings/ankh_large'
# INPUT_DIM = 1536 

# 2. ProtT5 XL (異붿쿇!)
# MODEL_NAME = "prott5_xl"
# EMBEDDING_DIR = './data/embeddings/protT5_xl' # ?대뜑紐??뺤씤 ?꾩슂
# INPUT_DIM = 1024

# 3. ESM2-15B (湲곗〈 ?깃났 紐⑤뜽 蹂듦뎄??
MODEL_NAME = "esm2_15B"
EMBEDDING_DIR = './data/embeddings/esm2_15B'
INPUT_DIM = 5120

# =========================================================
# ?뱛 怨듯넻 寃쎈줈 (?섏젙 遺덊븘??
# =========================================================
TRAIN_EMB_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_emb.npy')
TRAIN_IDS_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_ids.npy')
TEST_EMB_FILE  = os.path.join(EMBEDDING_DIR, 'testsuperset_emb.npy')
TEST_IDS_FILE  = os.path.join(EMBEDDING_DIR, 'testsuperset_ids.npy')
TRAIN_TERMS = './data/raw/train/train_terms.tsv'
OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 64
LR = 1e-3  # ?뵦 以묒슂: 1e-4?먯꽌 1e-3?쇰줈 蹂듦뎄 (Saturation 諛⑹?)
EPOCHS = 8
SAVE_THRESHOLD = 0.01  
TOP_K = 150 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# ?룛截?Dataset & Model (PPI ?놁쓬, ?쒖젙 ?곹깭)
# =========================================================
class AggregatedDataset(Dataset):
    def __init__(self, emb_matrix, id_map, target_pids, labels_dict=None, num_classes=0):
        self.emb_matrix = emb_matrix
        self.id_map = id_map
        self.target_pids = target_pids
        self.labels_dict = labels_dict
        self.num_classes = num_classes
    def __len__(self): return len(self.target_pids)
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
        if isinstance(pid, bytes): pid = pid.decode('utf-8')
        pid = str(pid).strip().replace('>', '')
        if '|' in pid: parts = pid.split('|'); pid = parts[1] if len(parts) >= 2 else pid
        id_list.append(pid)
    return id_list

def train_cycle(name, criterion, output_path, train_loader, test_loader, num_classes):
    print(f"\n?? [{name}] Training...")
    model = SimpleMLP(INPUT_DIM, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        model.train()
        for x, y in tqdm(train_loader, desc=f"{name} Ep {epoch+1}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    print(f"?뵰 [{name}] Inference...")
    model.eval()
    with open(output_path, 'w') as f:
        with torch.no_grad():
            for x, pids in tqdm(test_loader, desc=f"{name} Pred"):
                x = x.to(device)
                probs = torch.sigmoid(model(x)).cpu().numpy()
                for i, pid in enumerate(pids):
                    p = probs[i]
                    indices = np.where(p >= SAVE_THRESHOLD)[0]
                    scores = p[indices]
                    # Saturation Check
                    if len(scores) > 0 and abs(scores[0] - 0.5) < 0.01:
                        # 0.5 洹쇱쿂硫?嫄대꼫?곌린 (?덉쟾?μ튂)
                        continue
                    if len(scores) > TOP_K:
                        top_idx = np.argsort(scores)[-TOP_K:]
                        indices = indices[top_idx]
                        scores = scores[top_idx]
                    for idx, score in zip(indices, scores):
                        f.write(f"{pid}\t{idx2term[idx]}\t{score:.5f}\n")
    print(f"??Saved: {output_path}")

# =========================================================
# ?뵦 Main Execution
# =========================================================
if __name__ == "__main__":
    print(f"?뵦 [{MODEL_NAME}] Training Start... (Dim: {INPUT_DIM})")
    
    train_emb = np.load(TRAIN_EMB_FILE, mmap_mode='r')
    test_emb = np.load(TEST_EMB_FILE, mmap_mode='r')
    train_ids = clean_ids(np.load(TRAIN_IDS_FILE, allow_pickle=True))
    test_ids = clean_ids(np.load(TEST_IDS_FILE, allow_pickle=True))
    train_id_map = {pid: i for i, pid in enumerate(train_ids)}
    test_id_map = {pid: i for i, pid in enumerate(test_ids)}
    
    train_terms = pd.read_csv(TRAIN_TERMS, sep='\t')
    term_counts = train_terms['term'].value_counts()
    top_terms = term_counts[term_counts >= 10].index.tolist()
    term2idx = {t: i for i, t in enumerate(top_terms)}
    idx2term = {i: t for i, t in enumerate(top_terms)}
    
    train_labels = {}
    for pid, term in zip(train_terms['EntryID'], train_terms['term']):
        pid = str(pid).strip()
        if term in term2idx and pid in train_id_map:
            if pid not in train_labels: train_labels[pid] = []
            train_labels[pid].append(term2idx[term])
    
    valid_pids = list(train_labels.keys())
    trn_ids, _ = train_test_split(valid_pids, test_size=0.1, random_state=42)
    
    train_ds = AggregatedDataset(train_emb, train_id_map, trn_ids, train_labels, len(top_terms))
    test_ds  = AggregatedDataset(test_emb, test_id_map, test_ids, num_classes=0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 1. BCE
    bce_out = f"{OUTPUT_DIR}/pred_{MODEL_NAME}_BCE.tsv"
    train_cycle("BCE", nn.BCEWithLogitsLoss(), bce_out, train_loader, test_loader, len(top_terms))

    # 2. ASL
    asl_out = f"{OUTPUT_DIR}/pred_{MODEL_NAME}_ASL.tsv"
    train_cycle("ASL", AsymmetricLoss(), asl_out, train_loader, test_loader, len(top_terms))

    # 3. Merge (Score Average)
    print("\n?뵕 Merging...")
    df_bce = pd.read_csv(bce_out, sep='\t', names=['Id', 'Term', 'Score'], header=None)
    df_asl = pd.read_csv(asl_out, sep='\t', names=['Id', 'Term', 'Score'], header=None)
    merged = pd.merge(df_bce, df_asl, on=['Id', 'Term'], suffixes=('_bce', '_asl'), how='outer').fillna(0)
    
    merged['Score'] = (merged['Score_bce'] + merged['Score_asl']) / 2
    final_out = f"{OUTPUT_DIR}/pred_{MODEL_NAME}_FINAL.tsv"
    merged[['Id', 'Term', 'Score']].to_csv(final_out, sep='\t', index=False, header=False)
    
    print(f"?럦 Saved: {final_out}")
