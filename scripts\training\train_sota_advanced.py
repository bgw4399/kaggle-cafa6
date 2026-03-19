import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import obonet
import networkx as nx

# =========================================================
# ?숋툘 Config
# =========================================================
EMBEDDING_DIR = './data/embeddings/esm2_15B'
TRAIN_EMB_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_emb.npy')
TRAIN_IDS_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_ids.npy')
TEST_EMB_FILE = os.path.join(EMBEDDING_DIR, 'testsuperset_emb.npy') 
TEST_IDS_FILE = os.path.join(EMBEDDING_DIR, 'testsuperset_ids.npy') 

TRAIN_TERMS = './data/raw/train/train_terms.tsv'
GO_OBO = './data/raw/train/go-basic.obo'
OUTPUT_FILE = './results/sota_advanced/submission_SOTA_Advanced.tsv'
os.makedirs('./results/sota_advanced', exist_ok=True)

INPUT_DIM = 5120
LABEL_EMB_DIM = 1024 # Dimension for Label Embeddings
BATCH_SIZE = 128
LR = 5e-4 
EPOCHS = 10
LAMBDA_HIER = 0.05 # Weight for Hierarchy Regularization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"?뵦 Device: {device} | SOTA Advanced: Label Emb + Hierarchy Reg")

# =========================================================
# ?빖截?Hierarchy / Ontology Utils
# =========================================================
def build_hierarchy_mask(terms_list, obo_path):
    print("   ?빖截?building Hierarchy Mask from OBO...")
    graph = obonet.read_obo(obo_path)
    
    term_to_idx = {t: i for i, t in enumerate(terms_list)}
    num_classes = len(terms_list)
    
    # Create (Child, Parent) pairs indices
    # We want to penalize if Child > Parent
    # Mask: [N_edges, 2] -> (child_idx, parent_idx)
    edges = []
    
    for term in terms_list:
        if term not in graph: continue
        child_idx = term_to_idx[term]
        
        # Get parents (is_a relationship)
        # NetworkX: graph.predecessors(child) ? No, in obonet:
        # edge (u, v, key=is_a) means u IS_A v (u is child, v is parent)
        # Verify direction: obonet usually loads u->v as u is_a v.
        # So parents are successors? Let's check typical usage.
        # Standard: ancestors are reachable from child.
        
        for parent in graph.successors(term):
            if parent in term_to_idx:
                parent_idx = term_to_idx[parent]
                edges.append((child_idx, parent_idx))
                
    edges_tensor = torch.tensor(edges, dtype=torch.long).to(device)
    print(f"     Found {len(edges):,} constraints for {num_classes} terms.")
    return edges_tensor

# =========================================================
# ?쭬 Model: Label Embedding Network
# =========================================================
class LabelEmbeddingModel(nn.Module):
    def __init__(self, input_dim, num_classes, label_emb_dim=1024):
        super().__init__()
        
        # 1. Protein Encoder (Project ESM to Joint Space)
        self.protein_encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(2048, label_emb_dim), # Project to d
            nn.BatchNorm1d(label_emb_dim) # Normalize for dot product stability
        )
        
        # 2. Label Embeddings (Learnable)
        # Shape: [C, d]
        # Can be initialized with word2vec or node2vec if available, 
        # but random init + learning is also "TALE-like".
        self.label_embeddings = nn.Parameter(torch.randn(num_classes, label_emb_dim))
        nn.init.xavier_uniform_(self.label_embeddings)
        
    def forward(self, x):
        # x: [B, In]
        prot_emb = self.protein_encoder(x) # [B, d]
        prot_emb = F.normalize(prot_emb, p=2, dim=1) # Cosine Sim style
        
        lbl_emb = F.normalize(self.label_embeddings, p=2, dim=1) # [C, d]
        
        # Dot Product: [B, d] @ [d, C] -> [B, C]
        logits = torch.matmul(prot_emb, lbl_emb.t())
        
        # Scale logits (Cosine sim is [-1, 1], we need larger range for sigmoid)
        logits = logits * 10.0 
        
        return logits

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
    # 1. Data
    print("   ?뱿 Loading Embeddings...")
    train_emb = np.load(TRAIN_EMB_FILE, mmap_mode='r')
    train_ids = clean_ids(np.load(TRAIN_IDS_FILE, allow_pickle=True))
    train_id_map = {pid: i for i, pid in enumerate(train_ids)}
    
    # 2. Labels
    print("   ?뱿 Processing Labels...")
    df = pd.read_csv(TRAIN_TERMS, sep='\t')
    top_terms = df['term'].value_counts()
    top_terms = top_terms[top_terms >= 50].index.tolist()
    print(f"     Target Classes: {len(top_terms)}")
    
    term2idx = {t: i for i, t in enumerate(top_terms)}
    idx2term = {i: t for i, t in enumerate(top_terms)}
    
    labels = {}
    for pid, term in zip(df['EntryID'], df['term']):
        pid = str(pid)
        if term in term2idx:
            labels.setdefault(pid, []).append(term2idx[term])

    # 3. Hierarchy Constraints
    hier_edges = build_hierarchy_mask(top_terms, GO_OBO)
    
    # 4. Train Setup
    train_ds = AggregatedDataset(train_emb, train_id_map, train_ids, labels, len(top_terms))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = LabelEmbeddingModel(INPUT_DIM, len(top_terms), LABEL_EMB_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    bce_loss = nn.BCEWithLogitsLoss()
    
    print("\n?? Training Advanced SOTA Model...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_hier = 0
        
        for x, y in tqdm(train_loader, desc=f"Ep {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x)
            
            # Base Loss
            cls_loss = bce_loss(logits, y)
            
            # Hierarchy Loss
            # P(Child) > P(Parent) => Violation
            # We want Sigmoid(Child) <= Sigmoid(Parent)
            # Implies Logits(Child) <= Logits(Parent) roughly
            # Penalty = ReLU(Logits(Child) - Logits(Parent))
            
            # logits: [B, C]
            # hier_edges: [E, 2] -> (child, parent)
            child_scores = logits[:, hier_edges[:, 0]] # [B, E]
            parent_scores = logits[:, hier_edges[:, 1]] # [B, E]
            
            diff = child_scores - parent_scores
            penalty = torch.relu(diff).sum(dim=1).mean() # Sum over edges, Mean over batch
            
            loss = cls_loss + LAMBDA_HIER * penalty
            
            loss.backward()
            optimizer.step()
            
            total_loss += cls_loss.item()
            total_hier += penalty.item()
            
        print(f"   Ep {epoch+1} Loss: {total_loss/len(train_loader):.4f} | HierPenalty: {total_hier/len(train_loader):.4f}")

    # 5. Inference
    print("\n?뵰 Generating Test Predictions...")
    model.eval()
    
    if os.path.exists(TEST_EMB_FILE):
        test_emb = np.load(TEST_EMB_FILE, mmap_mode='r')
        try:
            test_ids = clean_ids(np.load(TEST_IDS_FILE, allow_pickle=True))
        except: pass
        
        test_ds = AggregatedDataset(test_emb, {pid: i for i, pid in enumerate(test_ids)}, test_ids, num_classes=0)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        f_out = open(OUTPUT_FILE, 'w')
        with torch.no_grad():
            for x, pids in tqdm(test_loader, desc="Inference"):
                x = x.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                for i, pid in enumerate(pids):
                    p = probs[i]
                    indices = np.where(p >= 0.01)[0]
                    scores = p[indices]
                    if len(scores) > 150:
                        top = np.argsort(scores)[-150:]
                        indices = indices[top]
                    
                    for idx in indices:
                        f_out.write(f"{pid}\t{idx2term[idx]}\t{p[idx]:.5f}\n")
        f_out.close()
        print(f"??Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()


