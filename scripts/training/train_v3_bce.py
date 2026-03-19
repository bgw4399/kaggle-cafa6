import os
import gc
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
# ?㎚ Ontology Helpers
# =========================================================
def load_obo_parents(path):
    print(f"?뱴 Loading OBO from {path}...")
    parents_map = defaultdict(set)
    namespaces = {}
    term = None
    with open(path, 'r', encoding='utf-8') as f:
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
            elif line.startswith('is_a: ') and term:
                parent = line[6:].split()[0]
                parents_map[term].add(parent)
            elif line.startswith('relationship: part_of ') and term:
                parent = line[22:].split()[0]
                parents_map[term].add(parent)
    print(f"   Loaded {len(parents_map)} terms, {len(namespaces)} namespaces.")
    return parents_map, namespaces

def propagate_labels(pid_terms, parents_map):
    print("?봽 Propagating labels...")
    new_labels = {}
    term_ancestors = {}
    
    def get_ancestors(t):
        if t in term_ancestors: return term_ancestors[t]
        ancestors = set()
        stack = [t]
        while stack:
            curr = stack.pop()
            if curr in parents_map:
                for p in parents_map[curr]:
                    if p not in ancestors:
                        ancestors.add(p)
                        stack.append(p)
        term_ancestors[t] = ancestors
        return ancestors

    for pid, terms in tqdm(pid_terms.items(), desc="Propagating"):
        expanded = set(terms)
        for t in terms:
            expanded.update(get_ancestors(t))
        new_labels[pid] = expanded
        
    return new_labels

# =========================================================
# ?숋툘 Config
# =========================================================
MODEL_NAME = "esm2_15B_v3"
EMBEDDING_DIR = './data/embeddings/esm2_15B'
INPUT_DIM = 5120

TRAIN_EMB_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_emb.npy')
TRAIN_IDS_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_ids.npy')
TEST_EMB_FILE  = os.path.join(EMBEDDING_DIR, 'testsuperset_emb.npy')
TEST_IDS_FILE  = os.path.join(EMBEDDING_DIR, 'testsuperset_ids.npy')

TRAIN_TERMS = './data/raw/train/train_terms.tsv'
OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 15
SAVE_THRESHOLD = 0.01
TOP_K = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"?뵦 [{MODEL_NAME}] Training with BCE + Namespace Weights...")

# =========================================================
# Dataset
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

# =========================================================
# Simple MLP (proven to work)
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

# =========================================================
# Weighted BCE Loss (BP gets more weight)
# =========================================================
class WeightedBCE(nn.Module):
    def __init__(self, term_weights):
        super().__init__()
        self.term_weights = term_weights
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, x, y):
        loss = self.bce(x, y)
        loss = loss * self.term_weights.unsqueeze(0)
        return loss.mean()

def clean_ids(id_array):
    id_list = []
    flat_array = id_array.reshape(-1)
    for pid in flat_array:
        if isinstance(pid, bytes): pid = pid.decode('utf-8')
        pid = str(pid).strip().replace('>', '')
        if '|' in pid: parts = pid.split('|'); pid = parts[1] if len(parts) >= 2 else pid
        id_list.append(pid)
    return id_list

# =========================================================
# Training
# =========================================================
def train_and_predict(name, model, criterion, output_file, train_loader, test_loader, epochs, idx2term):
    print(f"\n?? [{name}] Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"{name} Ep {epoch+1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        print(f"   ?뱟 Epoch {epoch+1}: Avg Loss = {avg_loss:.5f}")
    
    print(f"?뵰 [{name}] Inference...")
    model.eval()
    with open(output_file, 'w') as f:
        with torch.no_grad():
            for x, pids in tqdm(test_loader, desc="Inference"):
                x = x.to(device)
                probs = torch.sigmoid(model(x)).cpu().numpy()
                lines = []
                for i, pid in enumerate(pids):
                    p = probs[i]
                    indices = np.where(p >= SAVE_THRESHOLD)[0]
                    scores = p[indices]
                    if len(scores) > TOP_K:
                        top_idx = np.argsort(scores)[-TOP_K:]
                        indices = indices[top_idx]
                        scores = scores[top_idx]
                    for idx, score in zip(indices, scores):
                        lines.append(f"{pid}\t{idx2term[idx]}\t{score:.5f}\n")
                f.writelines(lines)
    print(f"??Saved: {output_file}")

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print("   Loading Data...")
    train_emb = np.load(TRAIN_EMB_FILE, mmap_mode='r')
    test_emb = np.load(TEST_EMB_FILE, mmap_mode='r')
    train_ids = clean_ids(np.load(TRAIN_IDS_FILE, allow_pickle=True))
    test_ids = clean_ids(np.load(TEST_IDS_FILE, allow_pickle=True))
    
    train_id_map = {pid: i for i, pid in enumerate(train_ids)}
    test_id_map = {pid: i for i, pid in enumerate(test_ids)}
    
    train_terms = pd.read_csv(TRAIN_TERMS, sep='\t')
    
    OBO_PATH = "./data/raw/train/go-basic.obo"
    parents_map, namespaces = load_obo_parents(OBO_PATH)
    
    raw_labels = defaultdict(set)
    for pid, term in zip(train_terms['EntryID'], train_terms['term']):
        raw_labels[str(pid).strip()].add(term)
         
    prop_labels_map = propagate_labels(raw_labels, parents_map)
    
    print("?뱤 Building term list...")
    all_terms = []
    for terms in prop_labels_map.values():
        all_terms.extend(terms)
        
    term_counts = pd.Series(all_terms).value_counts()
    top_terms = term_counts[term_counts >= 10].index.tolist()
    
    term2idx = {t: i for i, t in enumerate(top_terms)}
    idx2term = {i: t for i, t in enumerate(top_terms)}
    
    # Namespace weights: BP=1.5, MF=1.0, CC=0.8
    print("?뵩 Building namespace weights...")
    NS_WEIGHTS = {'BP': 1.5, 'MF': 1.0, 'CC': 0.8}
    term_weights = torch.ones(len(top_terms))
    for i, t in enumerate(top_terms):
        ns = namespaces.get(t, 'BP')
        term_weights[i] = NS_WEIGHTS.get(ns, 1.0)
    term_weights = term_weights.to(device)
    
    train_labels = {}
    valid_pids = []
    for pid, terms in prop_labels_map.items():
        if pid in train_id_map:
            indices = [term2idx[t] for t in terms if t in term2idx]
            if indices:
                train_labels[pid] = indices
                valid_pids.append(pid)
            
    trn_ids, _ = train_test_split(valid_pids, test_size=0.1, random_state=42)
    
    train_ds = AggregatedDataset(train_emb, train_id_map, trn_ids, train_labels, len(top_terms))
    test_ds  = AggregatedDataset(test_emb,  test_id_map,  test_ids, num_classes=0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n?뱤 Training: {len(trn_ids):,} proteins, {len(top_terms):,} GO terms")

    # Train with Weighted BCE
    out_file = f"{OUTPUT_DIR}/pred_{MODEL_NAME}_BCE.tsv"
    model = SimpleMLP(INPUT_DIM, len(top_terms)).to(device)
    criterion = WeightedBCE(term_weights)
    train_and_predict("WeightedBCE", model, criterion, out_file, train_loader, test_loader, EPOCHS, idx2term)

    print(f"\n?럦 Done: {out_file}")


