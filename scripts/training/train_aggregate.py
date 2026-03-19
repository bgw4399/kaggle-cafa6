import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from collections import defaultdict

# =========================================================
# ?㎚ Ontology Helpers (NEW)
# =========================================================
def load_obo_parents(path):
    print(f"?뱴 Loading OBO from {path}...")
    parents_map = defaultdict(set)
    term = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '[Term]':
                term = None
            elif line.startswith('id: '):
                term = line[4:].split()[0]
            elif line.startswith('is_a: ') and term:
                parent = line[6:].split()[0]
                parents_map[term].add(parent)
            elif line.startswith('relationship: part_of ') and term:
                parent = line[22:].split()[0]
                parents_map[term].add(parent)
    print(f"   Loaded {len(parents_map)} terms.")
    return parents_map

def propagate_labels(pid_terms, parents_map):
    """
    pid_terms: dict {pid: set(terms)}
    Returns: dict {pid: set(terms_with_ancestors)}
    """
    print("?봽 Propagating labels (Ancestor Inference)...")
    new_labels = {}
    
    # Memoization for term ancestors to speed up
    term_ancestors = {}
    
    def get_ancestors(t):
        if t in term_ancestors: return term_ancestors[t]
        
        ancestors = set()
        stack = [t]
        while stack:
            curr = stack.pop()
            # Parents
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
# ?숋툘 [?ㅼ젙] ?ш린留?二쇱꽍 ??댁꽌 ?곗꽭?? (?깃났 諛⑹젙??蹂듭젣)
# =========================================================

# [Case 1] Ankh Large (?ㅼ뼇??1?)
# MODEL_NAME = "ankh_large"
# EMBEDDING_DIR = './data/embeddings/ankh3_large'
# INPUT_DIM = 1536 

# [Case 2] ProtT5 XL (?ㅼ뼇??2?)
# MODEL_NAME = "prott5_xl"
# EMBEDDING_DIR = './data/embeddings/protT5_xl'
# INPUT_DIM = 1024

# [Case 3] ESM2-15B (SOTA 蹂듦뎄??
MODEL_NAME = "esm2_15B"
EMBEDDING_DIR = './data/embeddings/esm2_15B'
INPUT_DIM = 5120

# =========================================================
# ?뱛 怨듯넻 寃쎈줈 (SOTA 肄붾뱶 洹몃?濡??좎?)
# =========================================================
TRAIN_EMB_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_emb.npy')
TRAIN_IDS_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_ids.npy')
TEST_EMB_FILE  = os.path.join(EMBEDDING_DIR, 'testsuperset_emb.npy')
TEST_IDS_FILE  = os.path.join(EMBEDDING_DIR, 'testsuperset_ids.npy')

TRAIN_TERMS = './data/raw/train/train_terms.tsv'
OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ?섏씠?쇳뙆?쇰???(寃利앸맂 媛?
BATCH_SIZE = 64   # 15B???щ땲源?64, ?섎㉧吏??128 媛??LR = 1e-3         # SOTA LR ?좎?
EPOCHS = 8      # 10 epoch ?좎?
SAVE_THRESHOLD = 0.01
TOP_K = 100    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"?뵦 [{MODEL_NAME}] SOTA Logic Start... (Dim: {INPUT_DIM})")

# =========================================================
# ?룛截?Dataset & Model (洹몃?濡??좎?)
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
    # gamma_pos=1 ???듭떖 鍮꾧껐?댁뿀?듬땲?? (?덈? 0?쇰줈 諛붽씀吏 留덉꽭??
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x_sig = torch.sigmoid(x)
        xs_pos = x_sig
        xs_neg = 1 - x_sig

        # Probability Shifting
        if self.clip > 0: 
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Loss 怨꾩궛 (?쏅궇 ?섏떇 洹몃?濡?
        # gamma_pos=1 ?뺣텇???ъ슫 ?뺣떟???곷떦??臾댁떆?섏뿬 怨쇱쟻?⑹쓣 留됱븘以띾땲??
        loss = -1 * (y * torch.log(xs_pos.clamp(min=self.eps)) * (1 - xs_pos)**self.gamma_pos + 
                     (1 - y) * torch.log(xs_neg.clamp(min=self.eps)) * (1 - xs_neg)**self.gamma_neg)
        
        # 諛곗튂 ?ш린濡??섎닠二쇰뒗 寃껊쭔 ?딆? ?딆쑝硫??⑸땲??
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
# ?룂 ?숈뒿 ?ъ씠??(洹몃?濡??좎?)
# =========================================================
def train_and_predict(name, criterion, output_file, train_loader, test_loader, num_classes,EPOCHS):
    print(f"\n?? [{name}] Training...")
    model = SimpleMLP(INPUT_DIM, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    loss_history = [] # Loss 湲곕줉??    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Tqdm 諛붿뿉 Loss ?쒖떆
        pbar = tqdm(train_loader, desc=f"{name} Ep {epoch+1}/{EPOCHS}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # ?ㅼ떆媛?Loss ?쒖떆
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # ?먰룷??醫낅즺 ???됯퇏 Loss 異쒕젰
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
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
# ?뵦 硫붿씤 ?ㅽ뻾 (Rank Ensemble 濡쒖쭅 蹂듭젣)
# =========================================================
if __name__ == "__main__":
    # 1. Load Data
    print("   Loading Data...")
    train_emb = np.load(TRAIN_EMB_FILE, mmap_mode='r')
    test_emb = np.load(TEST_EMB_FILE, mmap_mode='r')
    
    # allow_pickle=True 異붽? (?덉쟾?μ튂)
    train_ids = clean_ids(np.load(TRAIN_IDS_FILE, allow_pickle=True))
    test_ids = clean_ids(np.load(TEST_IDS_FILE, allow_pickle=True))
    
    train_id_map = {pid: i for i, pid in enumerate(train_ids)}
    test_id_map = {pid: i for i, pid in enumerate(test_ids)}
    
    train_terms = pd.read_csv(TRAIN_TERMS, sep='\t')
    # 1-2. Load Ontology & Propagate
    OBO_PATH = "./data/raw/train/go-basic.obo"
    if os.path.exists(OBO_PATH):
        parents_map = load_obo_parents(OBO_PATH)
        
        # Build raw labels
        raw_labels = defaultdict(set)
        for pid, term in zip(train_terms['EntryID'], train_terms['term']):
             # clean pid?
             raw_labels[str(pid).strip()].add(term)
             
        # Propagate
        prop_labels_map = propagate_labels(raw_labels, parents_map)
        
        # Re-count terms based on PROPAGATED labels
        print("?뱤 Re-counting terms after propagation...")
        all_terms = []
        for terms in prop_labels_map.values():
            all_terms.extend(terms)
            
        term_counts = pd.Series(all_terms).value_counts()
    else:
        print("?좑툘 OBO File not found! Skipping propagation (NOT RECOMMENDED).")
        prop_labels_map = None # Fallback?

    top_terms = term_counts[term_counts >= 10].index.tolist()
    
    term2idx = {t: i for i, t in enumerate(top_terms)}
    idx2term = {i: t for i, t in enumerate(top_terms)}
    
    train_labels = {}
    # Use prop_labels_map if available
    source_map = prop_labels_map if prop_labels_map else defaultdict(list)
    
    # Needs to match train_id_map keys
    # source_map keys are strings. train_id_map keys are strings.
    
    valid_pids = []
    
    print("   Mapping Labels to Indices...")
    for pid, terms in source_map.items():
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

    # 2. BCE Training
    bce_out = f"{OUTPUT_DIR}/pred_{MODEL_NAME}_BCE.tsv"
    if not os.path.exists(bce_out):
        train_and_predict("BCE", nn.BCEWithLogitsLoss(), bce_out, train_loader, test_loader, len(top_terms),10)

    # 3. ASL Training
    asl_out = f"{OUTPUT_DIR}/pred_{MODEL_NAME}_ASL.tsv"
    if not os.path.exists(asl_out):
        asl_criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=0, clip=0.05)
        train_and_predict("ASL", asl_criterion, asl_out, train_loader, test_loader, len(top_terms), 15)

    # 4. ?뵦 Rank Ensemble (SOTA???듭떖!)
    print("\n?뵕 Rank Ensemble (The SOTA Logic)...")
    df_bce = pd.read_csv(bce_out, sep='\t', names=['Id', 'Term', 'Score'], header=None)
    df_asl = pd.read_csv(asl_out, sep='\t', names=['Id', 'Term', 'Score'], header=None)

    # Rank 蹂??    df_bce['Rank'] = df_bce['Score'].rank(pct=True)
    df_asl['Rank'] = df_asl['Score'].rank(pct=True)

    merged = pd.merge(df_bce, df_asl, on=['Id', 'Term'], suffixes=('_bce', '_asl'), how='outer').fillna(0)
    
    # Rank ?됯퇏
    merged['Score'] = (merged['Rank_bce'] + merged['Rank_asl']) / 2
    
    # ???    final_out = f"{OUTPUT_DIR}/pred_{MODEL_NAME}_FINAL.tsv"
    merged[['Id', 'Term', 'Score']].to_csv(final_out, sep='\t', index=False, header=False)

    print(f"?럦 {MODEL_NAME} ?숈뒿 ?꾨즺: {final_out}")
    print("?몛 ???뚯씪??諛붾줈 'Rank Ensemble'???곸슜??怨좏뭹吏??뚯씪?낅땲??")

