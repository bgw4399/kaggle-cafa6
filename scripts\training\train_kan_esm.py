import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import math

# =========================================================
# ?숋툘 Config
# =========================================================
# Using the path found in train_esm_scientific.py
EMBEDDING_DIR = './data/embeddings/esm2_15B'
TRAIN_EMB_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_emb.npy')
TRAIN_IDS_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_ids.npy')
TEST_EMB_FILE = os.path.join(EMBEDDING_DIR, 'testsuperset_emb.npy') # Corrected
TEST_IDS_FILE = os.path.join(EMBEDDING_DIR, 'testsuperset_ids.npy') 

TRAIN_TERMS = './data/raw/train/train_terms.tsv'
OUTPUT_FILE = './results/kan/submission_KAN_ESM.tsv'
os.makedirs('./results/kan', exist_ok=True)

INPUT_DIM = 5120
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 5 # KAN converges differently, start small
SAVE_THRESHOLD = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"?뵦 Device: {device} | SOTA Architecture: KAN (Kolmogorov-Arnold Network)")

# =========================================================
# ?쭬 KAN Layer Implementation (Efficient Version)
# =========================================================
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(2)
        grid = grid.unsqueeze(0)
        B = (x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])
        B = B.float()
        for k in range(1, self.spline_order + 1):
            B = (x - grid[:, :, : -(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, : -(k + 1)]) * B[:, :, :-1] + (
                grid[:, :, k + 1 :] - x
            ) / (grid[:, :, k + 1 :] - grid[:, :, 1:(-k)]) * B[:, :, 1:]
        return B

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8) # Simple Norm for Grid
        x_norm = x_norm * 2 - 1 # range [-1, 1]
        
        spline_output = F.linear(
            self.b_splines(x_norm).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

class KANClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # KAN Layer 1: Input -> Hidden
        self.kan1 = KANLinear(input_dim, 256, grid_size=3) # Reduced dim for speed
        self.ln1 = nn.LayerNorm(256)
        
        # KAN Layer 2: Hidden -> Output
        self.kan2 = KANLinear(256, num_classes, grid_size=3)
        
    def forward(self, x):
        x = self.kan1(x)
        x = self.ln1(x)
        x = self.kan2(x)
        return x

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
    
    # 2. Labels
    print("   ?뱿 Processing Labels...")
    df = pd.read_csv(TRAIN_TERMS, sep='\t')
    top_terms = df['term'].value_counts()
    top_terms = top_terms[top_terms >= 50].index.tolist() # Top 1500-2000 terms
    term2idx = {t: i for i, t in enumerate(top_terms)}
    idx2term = {i: t for i, t in enumerate(top_terms)}
    print(f"     Target Classes: {len(top_terms)}")
    
    labels = {}
    for pid, term in zip(df['EntryID'], df['term']):
        pid = str(pid)
        if term in term2idx:
            labels.setdefault(pid, []).append(term2idx[term])
            
    # 3. Train
    train_ds = AggregatedDataset(train_emb, train_id_map, train_ids, labels, len(top_terms))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = KANClassifier(INPUT_DIM, len(top_terms)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    print("\n?? Training KAN...")
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
        print(f"   Loss: {total/len(train_loader):.4f}")
        
    # 4. Inference
    print("\n?뵰 Generating Test Predictions...")
    model.eval()
    
    # Load Test Data (Lazy Load)
    if not os.path.exists(TEST_EMB_FILE):
        print(f"??Test Embeddings not found at {TEST_EMB_FILE}. Skipping Inference.")
        return

    print("   ?뱿 Loading Test Embeddings...")
    test_emb = np.load(TEST_EMB_FILE, mmap_mode='r')
    try:
        test_ids = clean_ids(np.load(TEST_IDS_FILE, allow_pickle=True))
    except:
        # Fallback if IDs are missing or different format
        print("   ?좑툘 Loading IDs from text file or alternative source...")
        # Placeholder: assume external ID file or fail gracefully
        pass
    
    # Create Test Dataset
    # We can reuse AggregatedDataset but without labels
    test_ds = AggregatedDataset(test_emb, {pid: i for i, pid in enumerate(test_ids)} if 'test_ids' in locals() else {}, 
                                test_ids if 'test_ids' in locals() else [], num_classes=0)
    
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    f_out = open(OUTPUT_FILE, 'w')
    
    with torch.no_grad():
        for x, pids in tqdm(test_loader, desc="Inference"):
            x = x.to(device)
            # Forward
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            for i, pid in enumerate(pids):
                p = probs[i]
                # Filter low scores to save space
                indices = np.where(p >= SAVE_THRESHOLD)[0]
                
                # Keep Top 100 per protein
                if len(indices) > 100:
                    sub_scores = p[indices]
                    top_k_idx = np.argsort(sub_scores)[-100:]
                    indices = indices[top_k_idx]
                
                for idx in indices:
                    score = p[idx]
                    term = idx2term[idx]
                    f_out.write(f"{pid}\t{term}\t{score:.5f}\n")
                    
    f_out.close()
    print(f"??Submission Saved: {OUTPUT_FILE}")
    
if __name__ == "__main__":
    main()


