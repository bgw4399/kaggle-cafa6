import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# =========================================================
# ?숋툘 Scientific Retraining Config (ProtT5)
# =========================================================
EMBEDDING_DIR = './data/embeddings/protT5_xl'
TRAIN_EMB_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_emb.npy')
TRAIN_IDS_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_ids.npy')

# Split Paths
SPLIT_DIR = './data/raw/train/Split'
TRAIN_TERMS = os.path.join(SPLIT_DIR, 'train_terms_split.tsv')
VAL_IDS_NPY = os.path.join(SPLIT_DIR, 'val_ids.npy')
TRAIN_IDS_NPY = os.path.join(SPLIT_DIR, 'train_ids.npy')

OUTPUT_DIR = './results/scientific/prott5'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparams
INPUT_DIM = 1024 
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 10
NUM_WORKERS = 4
SAVE_THRESHOLD = 0.01 
TOP_K = 150 # Generous Top-K for validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"?뵦 Device: {device} | Scientific Training: ProtT5")

# =========================================================
# ?룛截?Dataset
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
        # PID might be loaded as bytes or str, ensure str
        pid_str = str(pid)
        
        if pid_str in self.id_map:
            row_idx = self.id_map[pid_str]
            emb = self.emb_matrix[row_idx]
        else:
            # Fallback or error? For now zero
            emb = np.zeros(INPUT_DIM, dtype=np.float32)
            
        emb = torch.tensor(emb, dtype=torch.float32)

        if self.labels_dict is not None:
            label = np.zeros(self.num_classes, dtype=np.float32)
            if pid_str in self.labels_dict:
                for t_idx in self.labels_dict[pid_str]: label[t_idx] = 1.0
            return emb, torch.tensor(label)
        else:
            return emb, pid

# =========================================================
# ?쭬 Model
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

# =========================================================
# ?봽 Main
# =========================================================
def main():
    print("   ?뱿 Loading Embeddings...")
    train_emb = np.load(TRAIN_EMB_FILE, mmap_mode='r')
    
    # ID Mapping
    train_ids_raw = np.load(TRAIN_IDS_FILE, allow_pickle=True)
    train_ids = clean_ids(train_ids_raw)
    train_id_map = {pid: i for i, pid in enumerate(train_ids)}
    
    # Load Split IDs
    print("   ?뱿 Loading Splits...")
    trn_ids_raw = np.load(TRAIN_IDS_NPY, allow_pickle=True)
    val_ids_raw = np.load(VAL_IDS_NPY, allow_pickle=True)
    
    # Ensure list of strings
    trn_ids = [str(x) for x in trn_ids_raw]
    val_ids = [str(x) for x in val_ids_raw]
    
    print(f"     Train: {len(trn_ids):,}, Val: {len(val_ids):,}")
    
    # Prepare Labels (ONLY for Train Split)
    print("   ?뱿 Loading Train Terms...")
    train_terms = pd.read_csv(TRAIN_TERMS, sep='\t')
    term_counts = train_terms['term'].value_counts()
    top_terms = term_counts[term_counts >= 10].index.tolist()
    print(f"     Targets: {len(top_terms)}")
    
    term2idx = {t: i for i, t in enumerate(top_terms)}
    idx2term = {i: t for i, t in enumerate(top_terms)}
    
    train_labels = {}
    for pid, term in zip(train_terms['EntryID'], train_terms['term']):
        pid = str(pid).strip()
        if term in term2idx:
            if pid not in train_labels: train_labels[pid] = []
            train_labels[pid].append(term2idx[term])
            
    # Datasets
    train_ds = AggregatedDataset(train_emb, train_id_map, trn_ids, train_labels, len(top_terms))
    # For Validation, we don't need labels in Dataset if we just want to predict (Evaluator has GT)
    # BUT! To calculate Val Loss during training, we might want them.
    # However, user wants F-max. So we will just predict and dump to file.
    val_ds = AggregatedDataset(train_emb, train_id_map, val_ids, num_classes=0) # Inference Mode
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Model
    model = SimpleMLP(INPUT_DIM, len(top_terms)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Training Loop
    print("\n?? Training Start...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"   Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
        
    # Validation Inference
    print("\n?뵰 Generating Validation Predictions...")
    model.eval()
    val_out_file = os.path.join(OUTPUT_DIR, "val_pred_prott5.tsv")
    
    with open(val_out_file, 'w') as f:
        with torch.no_grad():
            for x, pids in tqdm(val_loader, desc="Inference"):
                x = x.to(device)
                probs = torch.sigmoid(model(x)).cpu().numpy()
                
                for i, pid in enumerate(pids):
                    # Filter and write sparse
                    p = probs[i]
                    # Simple Threshold + Top-K
                    indices = np.where(p >= SAVE_THRESHOLD)[0]
                    scores = p[indices]
                    
                    if len(scores) > TOP_K:
                        top_idx = np.argsort(scores)[-TOP_K:]
                        indices = indices[top_idx]
                        scores = scores[top_idx]
                        
                    for idx, score in zip(indices, scores):
                        f.write(f"{pid}\t{idx2term[idx]}\t{score:.5f}\n")
                        
    print(f"??Prediction Saved: {val_out_file}")

if __name__ == "__main__":
    main()


