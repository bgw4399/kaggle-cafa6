import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# =========================================================
# ?숋툘 Config
# =========================================================
# Paths (Reusing existing verified paths)
EMBEDDING_DIR = './data/embeddings/esm2_15B'
TRAIN_EMB_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_emb.npy')
TRAIN_IDS_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_ids.npy')
TEST_EMB_FILE = os.path.join(EMBEDDING_DIR, 'testsuperset_emb.npy') 
TEST_IDS_FILE = os.path.join(EMBEDDING_DIR, 'testsuperset_ids.npy') 
TRAIN_TERMS = './data/raw/train/train_terms.tsv'

OUTPUT_FILE = './results/pu_learning/submission_PU_ESM.tsv'
os.makedirs('./results/pu_learning', exist_ok=True)

INPUT_DIM = 5120
BATCH_SIZE = 128
LR = 1e-4 # PU learning can be unstable, lower LR
EPOCHS = 10
SAVE_THRESHOLD = 0.01
TOP_K = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"?뵦 Device: {device} | SOTA Strategy: PU Learning (Positive-Unlabeled)")

# =========================================================
# ?뱣 PU Loss (User Provided + Vectorized)
# =========================================================
class MultiLabelNNPULoss(nn.Module):
    """
    Multi-label non-negative PU loss.
    """
    def __init__(self, priors: torch.Tensor, reduction="mean"):
        super().__init__()
        # priors: [C] - Class prior probability (Frequency in dataset)
        self.register_buffer("priors", priors.float().clamp(1e-6, 1-1e-6))
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], y: [B, C] {0,1}
        y = y.float()
        pri = self.priors # [C]

        # 1. Sigmoid Losses
        # Positive Risk: Loss on Positive Samples (y=1)
        # L_pos = -log(sigmoid(f)) = softplus(-f)
        l_pos = F.softplus(-logits)
        
        # Negative Risk: Loss on All Samples assumed Negative
        # L_neg = -log(1-sigmoid(f)) = -log(sigmoid(-f)) = softplus(f)
        l_neg = F.softplus(logits)

        # 2. Risk Estimation
        # R_p^+(g) = (1/N_p) * sum(l_pos * y)
        pos_cnt = y.sum(dim=0).clamp(min=1.0) # [C] number of true positives per class
        Ep_lpos = (l_pos * y).sum(dim=0) / pos_cnt 
        
        # R_u^-(g) = (1/N_u) * sum(l_neg * (1-y)) -- Unlabeled risk
        # Note: Standard PU uses total expectation, but here we use unlabeled part
        unl = 1.0 - y
        unl_cnt = unl.sum(dim=0).clamp(min=1.0)
        Eu_lneg = (l_neg * unl).sum(dim=0) / unl_cnt
        
        # R_p^-(g) = (1/N_p) * sum(l_neg * y) -- Negative risk on positives (balancing term)
        Ep_lneg = (l_neg * y).sum(dim=0) / pos_cnt

        # 3. Non-Negative Correction
        # Risk = pi * R_p^+ + max(0, R_u^- - pi * R_p^-)
        # Note: User's formulation: Rp = pri * Ep_lpos
        # But usually Rp is just Ep_lpos (since we have labels).
        # Standard formulation: Risk = pi*R_p + max(0, R_x - pi*R_p) where R_x is risk on all data.
        # However, let's follow the user's snippet logic which balances by prior explicitly.
        
        Rp = pri * Ep_lpos
        Rn = Eu_lneg - pri * Ep_lneg
        
        loss = Rp + torch.relu(Rn) # [C]

        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()

# =========================================================
# ?쭬 Model (Standard MLP)
# =========================================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x): return self.layers(x)

# =========================================================
# ?룛截?Dataset & Utils
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
        pid_str = str(pid)
        if pid_str in self.id_map:
            row_idx = self.id_map[pid_str]
            emb = self.emb_matrix[row_idx]
        else:
            emb = np.zeros(INPUT_DIM, dtype=np.float32)
        emb = torch.tensor(emb, dtype=torch.float32)
        if self.labels_dict:
            label = np.zeros(self.num_classes, dtype=np.float32)
            if pid_str in self.labels_dict:
                for t in self.labels_dict[pid_str]: label[t] = 1.0
            return emb, torch.tensor(label)
        return emb, pid

def clean_ids(id_array):
    return [str(x).strip().replace('>','').split('|')[1] if '|' in str(x) else str(x).strip() for x in id_array.reshape(-1)]

def main():
    # 1. Load Data
    print("   ?뱿 Loading Embeddings...")
    train_emb = np.load(TRAIN_EMB_FILE, mmap_mode='r')
    train_ids = clean_ids(np.load(TRAIN_IDS_FILE, allow_pickle=True))
    train_id_map = {pid: i for i, pid in enumerate(train_ids)}
    
    # 2. Labels & Priors
    print("   ?뱿 Processing Labels & Calculating Priors...")
    df = pd.read_csv(TRAIN_TERMS, sep='\t')
    term_counts = df['term'].value_counts()
    
    # Filter common terms for stability
    top_terms = term_counts[term_counts >= 50].index.tolist()
    print(f"     Target Classes: {len(top_terms)}")
    
    term2idx = {t: i for i, t in enumerate(top_terms)}
    idx2term = {i: t for i, t in enumerate(top_terms)}
    
    # Calculate Priors (Frequency)
    total_samples = df['EntryID'].nunique()
    # priors[i] = count(term_i) / total_samples
    priors_vec = torch.zeros(len(top_terms))
    for i, term in enumerate(top_terms):
        cnt = term_counts[term]
        priors_vec[i] = cnt / total_samples
    
    print(f"     Priors Range: {priors_vec.min().item():.6f} ~ {priors_vec.max().item():.6f}")
    
    labels = {}
    for pid, term in zip(df['EntryID'], df['term']):
        pid = str(pid)
        if term in term2idx:
            labels.setdefault(pid, []).append(term2idx[term])
            
    # 3. Train
    train_ds = AggregatedDataset(train_emb, train_id_map, train_ids, labels, len(top_terms))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = SimpleMLP(INPUT_DIM, len(top_terms)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Init PU Loss
    criterion = MultiLabelNNPULoss(priors=priors_vec.to(device))
    
    print("\n?? Training with PU Learning...")
    for epoch in range(EPOCHS):
        model.train()
        total = 0
        for x, y in tqdm(train_loader, desc=f"Ep {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"   Ep {epoch+1} Loss: {total/len(train_loader):.4f}")
        
    # 4. Inference
    print("\n?뵰 Generating Test Predictions (PU)...")
    model.eval()
    
    if not os.path.exists(TEST_EMB_FILE):
        print(f"??Test Embeddings not found at {TEST_EMB_FILE}")
        return

    test_emb = np.load(TEST_EMB_FILE, mmap_mode='r')
    try:
        test_ids = clean_ids(np.load(TEST_IDS_FILE, allow_pickle=True))
    except:
        pass
    
    test_ds = AggregatedDataset(test_emb, {pid: i for i, pid in enumerate(test_ids)} if 'test_ids' in locals() else {}, 
                                test_ids if 'test_ids' in locals() else [], num_classes=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    f_out = open(OUTPUT_FILE, 'w')
    
    with torch.no_grad():
        for x, pids in tqdm(test_loader, desc="Inference"):
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            for i, pid in enumerate(pids):
                p = probs[i]
                indices = np.where(p >= SAVE_THRESHOLD)[0]
                
                if len(indices) > TOP_K:
                    sub_scores = p[indices]
                    top_k_idx = np.argsort(sub_scores)[-TOP_K:]
                    indices = indices[top_k_idx]
                
                for idx in indices:
                    f_out.write(f"{pid}\t{idx2term[idx]}\t{p[idx]:.5f}\n")
                    
    f_out.close()
    print(f"??Submission Saved: {OUTPUT_FILE}")
    
if __name__ == "__main__":
    main()


