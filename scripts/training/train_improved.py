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
    namespaces = {}  # NEW: Track namespace for each term
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
    print("?봽 Propagating labels (Ancestor Inference)...")
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
# ?숋툘 ?ㅼ젙
# =========================================================
MODEL_NAME = "esm2_15B_v2"  # NEW version tag
EMBEDDING_DIR = './data/embeddings/esm2_15B'
INPUT_DIM = 5120

TRAIN_EMB_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_emb.npy')
TRAIN_IDS_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_ids.npy')
TEST_EMB_FILE  = os.path.join(EMBEDDING_DIR, 'testsuperset_emb.npy')
TEST_IDS_FILE  = os.path.join(EMBEDDING_DIR, 'testsuperset_ids.npy')

TRAIN_TERMS = './data/raw/train/train_terms.tsv'
OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# ?뵩 IMPROVED Hyperparameters 
# =========================================================
BATCH_SIZE = 64
LR = 5e-4           # Slightly lower for stability with deeper net
EPOCHS_BCE = 10     # More BCE epochs
EPOCHS_ASL = 20     # More ASL epochs for rare term learning
SAVE_THRESHOLD = 0.01
TOP_K = 150         # Slightly higher to capture more BP terms

# Namespace weights for loss (BP gets more attention)
NS_WEIGHTS = {'BP': 1.5, 'MF': 1.0, 'CC': 0.8}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"?뵦 [{MODEL_NAME}] Improved Training Start... (Dim: {INPUT_DIM})")

# =========================================================
# ?룛截?IMPROVED Dataset & Model 
# =========================================================
class AggregatedDataset(Dataset):
    def __init__(self, emb_matrix, id_map, target_pids, labels_dict=None, num_classes=0, ns_weights=None, term_ns_idx=None):
        self.emb_matrix = emb_matrix
        self.id_map = id_map
        self.target_pids = target_pids
        self.labels_dict = labels_dict
        self.num_classes = num_classes
        self.ns_weights = ns_weights  # NEW: namespace weights per term
        self.term_ns_idx = term_ns_idx  # NEW: term -> namespace index
        
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
# ?쭬 IMPROVED MLP with Residual Connections
# =========================================================
class DeepResidualMLP(nn.Module):
    """Deeper MLP with residual connections for better BP learning"""
    def __init__(self, input_dim, num_classes, hidden_dim=1024):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        # Residual Block 1
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Residual Block 2 (narrower)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 2)
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Output
        self.fc_out = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Input projection
        out = F.relu(self.input_bn(self.input_proj(x)))
        
        # Residual Block 1
        identity = out
        out = F.relu(self.bn1(self.fc1(out)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = F.relu(out + identity)  # Residual connection
        
        # Residual Block 2
        identity = self.residual_proj(out)
        out = F.relu(self.bn3(self.fc3(out)))
        out = self.dropout(out)
        out = self.bn4(self.fc4(out))
        out = F.relu(out + identity)  # Residual connection
        
        return self.fc_out(out)

# =========================================================
# ?뱣 IMPROVED ASL Loss (Tuned for BP) - FIXED
# =========================================================
class ImprovedASL(nn.Module):
    """ASL with namespace-aware weighting - FIXED formula"""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, term_weights=None):
        super().__init__()
        self.gamma_neg = gamma_neg  # Higher = stronger negative suppression
        self.gamma_pos = gamma_pos  # >0 helps with hard positives
        self.clip = clip
        self.eps = eps
        self.term_weights = term_weights  # Per-term weights (BP: 1.5, etc.)

    def forward(self, x, y):
        x_sig = torch.sigmoid(x)
        
        # Positive loss (focal-style downweight easy positives)
        pt_pos = x_sig
        loss_pos = y * ((1 - pt_pos) ** self.gamma_pos) * torch.log(pt_pos.clamp(min=self.eps))
        
        # Negative loss with probability shifting (asymmetric)
        pt_neg = 1 - x_sig
        if self.clip > 0:
            # Shift probability for negatives (makes model less confident on negatives)
            pt_neg = (pt_neg + self.clip).clamp(max=1)
        # Focal-style downweight easy negatives (high p_neg = confident negative)
        loss_neg = (1 - y) * (pt_neg ** self.gamma_neg) * torch.log(pt_neg.clamp(min=self.eps))
        
        loss = -loss_pos - loss_neg
        
        # Apply term weights if available
        if self.term_weights is not None:
            loss = loss * self.term_weights.unsqueeze(0)
        
        return loss.sum() / x.size(0)

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
# ?룂 Training Loop (with logging)
# =========================================================
def train_and_predict(name, model, criterion, output_file, train_loader, test_loader, num_classes, epochs, idx2term):
    print(f"\n?? [{name}] Training with {type(model).__name__}...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"{name} Ep {epoch+1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"   ?뱟 Epoch {epoch+1}: Avg Loss = {avg_loss:.5f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
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
    return model

# =========================================================
# ?뵦 Main Execution
# =========================================================
if __name__ == "__main__":
    # 1. Load Data
    print("   Loading Data...")
    train_emb = np.load(TRAIN_EMB_FILE, mmap_mode='r')
    test_emb = np.load(TEST_EMB_FILE, mmap_mode='r')
    train_ids = clean_ids(np.load(TRAIN_IDS_FILE, allow_pickle=True))
    test_ids = clean_ids(np.load(TEST_IDS_FILE, allow_pickle=True))
    
    train_id_map = {pid: i for i, pid in enumerate(train_ids)}
    test_id_map = {pid: i for i, pid in enumerate(test_ids)}
    
    train_terms = pd.read_csv(TRAIN_TERMS, sep='\t')
    
    # 1-2. Load Ontology & Propagate
    OBO_PATH = "./data/raw/train/go-basic.obo"
    parents_map, namespaces = load_obo_parents(OBO_PATH)
    
    # Build raw labels
    raw_labels = defaultdict(set)
    for pid, term in zip(train_terms['EntryID'], train_terms['term']):
        raw_labels[str(pid).strip()].add(term)
         
    # Propagate
    prop_labels_map = propagate_labels(raw_labels, parents_map)
    
    # Re-count terms based on PROPAGATED labels
    print("?뱤 Re-counting terms after propagation...")
    all_terms = []
    for terms in prop_labels_map.values():
        all_terms.extend(terms)
        
    term_counts = pd.Series(all_terms).value_counts()
    top_terms = term_counts[term_counts >= 10].index.tolist()
    
    term2idx = {t: i for i, t in enumerate(top_terms)}
    idx2term = {i: t for i, t in enumerate(top_terms)}
    
    # Build namespace weights per term
    print("?뵩 Building namespace weights...")
    term_weights = torch.ones(len(top_terms))
    ns_counts = {'BP': 0, 'MF': 0, 'CC': 0}
    for i, t in enumerate(top_terms):
        ns = namespaces.get(t, 'BP')
        term_weights[i] = NS_WEIGHTS.get(ns, 1.0)
        ns_counts[ns] = ns_counts.get(ns, 0) + 1
    term_weights = term_weights.to(device)
    print(f"   NS Distribution: BP={ns_counts['BP']}, MF={ns_counts['MF']}, CC={ns_counts['CC']}")
    
    # Build train labels
    train_labels = {}
    valid_pids = []
    print("   Mapping Labels to Indices...")
    for pid, terms in prop_labels_map.items():
        if pid in train_id_map:
            indices = [term2idx[t] for t in terms if t in term2idx]
            if indices:
                train_labels[pid] = indices
                valid_pids.append(pid)
            
    trn_ids, val_ids = train_test_split(valid_pids, test_size=0.1, random_state=42)
    
    train_ds = AggregatedDataset(train_emb, train_id_map, trn_ids, train_labels, len(top_terms))
    test_ds  = AggregatedDataset(test_emb,  test_id_map,  test_ids, num_classes=0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n?뱤 Training Data: {len(trn_ids):,} proteins, {len(top_terms):,} GO terms")

    # 2. Train with Improved ASL + DeepResidualMLP
    asl_out = f"{OUTPUT_DIR}/pred_{MODEL_NAME}_ASL.tsv"
    model = DeepResidualMLP(INPUT_DIM, len(top_terms)).to(device)
    criterion = ImprovedASL(gamma_neg=6, gamma_pos=1, clip=0.05, term_weights=term_weights)
    train_and_predict("ASL_v2", model, criterion, asl_out, train_loader, test_loader, len(top_terms), EPOCHS_ASL, idx2term)

    print(f"\n?럦 {MODEL_NAME} Training Complete: {asl_out}")
    print("?몛 Next: Run final_merge_v2.py with this output file.")


