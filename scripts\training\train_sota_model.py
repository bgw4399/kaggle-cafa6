import os
import gc
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from collections import defaultdict

# =========================================================
# ?㎚ SOTA Architecture: Residual MLP
# =========================================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Skip connection
        return self.relu(out)

class ResMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=2048, num_blocks=2):
        super().__init__()
        # Project input to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Residual Blocks (Deep processing)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # Output Head
        self.head = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

# =========================================================
# ?뱣 Robust Loss: Focal Loss
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', term_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.term_weights = term_weights

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt is the probability of being classified correctly
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.term_weights is not None:
             focal_loss = focal_loss * self.term_weights.unsqueeze(0)

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

# =========================================================
# ?썱截?Helpers
# =========================================================
def clean_ids(id_array):
    id_list = []
    flat_array = id_array.reshape(-1)
    for pid in flat_array:
        if isinstance(pid, bytes): pid = pid.decode('utf-8')
        pid = str(pid).strip().replace('>', '')
        if '|' in pid: parts = pid.split('|'); pid = parts[1] if len(parts) >= 2 else pid
        id_list.append(pid)
    return id_list

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
            emb = np.zeros(self.emb_matrix.shape[1], dtype=np.float32)
        emb = torch.tensor(emb, dtype=torch.float32)
        
        if self.labels_dict:
            label = np.zeros(self.num_classes, dtype=np.float32)
            if pid in self.labels_dict:
                for t_idx in self.labels_dict[pid]: label[t_idx] = 1.0
            return emb, torch.tensor(label)
        return emb, pid

# =========================================================
# ?룂 Training Logic
# =========================================================
def train(args):
    print(f"\n?뵦 Training SOTA Model: {args.model_name}")
    print(f"   Embedding: {args.emb_dir}")
    
    # 1. Load Data
    train_emb = np.load(os.path.join(args.emb_dir, 'train_sequences_emb.npy'), mmap_mode='r')
    test_emb = np.load(os.path.join(args.emb_dir, 'testsuperset_emb.npy'), mmap_mode='r')
    
    train_ids = clean_ids(np.load(os.path.join(args.emb_dir, 'train_sequences_ids.npy')))
    test_ids = clean_ids(np.load(os.path.join(args.emb_dir, 'testsuperset_ids.npy')))
    
    train_id_map = {pid: i for i, pid in enumerate(train_ids)}
    test_id_map = {pid: i for i, pid in enumerate(test_ids)}
    
    input_dim = train_emb.shape[1]
    print(f"   Input Dim: {input_dim}")

    # 2. Terms & Labels (Common logic)
    print("   Loading Ontology & Terms...")
    train_terms = pd.read_csv("./data/raw/train/train_terms.tsv", sep='\t')
    term_counts = train_terms['term'].value_counts()
    top_terms = term_counts[term_counts >= 10].index.tolist() # Top 10 filter
    
    term2idx = {t: i for i, t in enumerate(top_terms)}
    idx2term = {i: t for i, t in enumerate(top_terms)}
    
    train_labels = defaultdict(list)
    for pid, term in zip(train_terms['EntryID'], train_terms['term']):
        if term in term2idx:
            train_labels[str(pid).strip()].append(term2idx[term])
            
    valid_pids = [pid for pid in train_labels.keys() if pid in train_id_map]
    trn_ids, val_ids = train_test_split(valid_pids, test_size=0.1, random_state=42)
    
    # 3. Load OBO for NS Weights
    ns_weights = torch.ones(len(top_terms))
    # (Simple logic: skip OBO parsing for speed if not essential, OR use fixed weights)
    # Using fixed weights for now based on previous success
    # BP terms are usually dominant, so we trust FocalLoss to handle it.
    
    # 4. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = AggregatedDataset(train_emb, train_id_map, trn_ids, train_labels, len(top_terms))
    val_ds = AggregatedDataset(train_emb, train_id_map, val_ids, train_labels, len(top_terms))
    test_ds = AggregatedDataset(test_emb, test_id_map, test_ids, num_classes=0)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    
    # 5. Model
    model = ResMLP(input_dim, len(top_terms), hidden_dim=2048, num_blocks=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = FocalLoss(gamma=2.0).to(device)
    
    # 6. Train Loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"   ?뱟 Ep {epoch+1}: Loss={avg_loss:.4f} LR={scheduler.get_last_lr()[0]:.6f}")
        
    # 7. Inference
    print("   ?뵰 Inference...")
    model.eval()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        with torch.no_grad():
            for x, pids in tqdm(test_loader):
                x = x.to(device)
                probs = torch.sigmoid(model(x)).cpu().numpy()
                for i, pid in enumerate(pids):
                    p = probs[i]
                    indices = np.where(p >= 0.01)[0]
                    scores = p[indices]
                    # Top-200
                    if len(scores) > 200:
                        top_idx = np.argsort(scores)[-200:]
                        indices = indices[top_idx]
                        scores = scores[top_idx]
                    for idx, score in zip(indices, scores):
                        f.write(f"{pid}\t{idx2term[idx]}\t{score:.5f}\n")
                        
    print(f"??Saved: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--emb_dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()
    
    train(args)

