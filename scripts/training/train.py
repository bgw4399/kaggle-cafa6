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
# ?숋툘 [?ㅼ젙] ProtT5_XL 留욎땄 ?ㅼ젙
# =========================================================
MODEL_NAME = "esm2_15B" 
EMBEDDING_DIR = './data/embeddings/esm2_15B' # ?대뜑紐??뺤씤 ?꾩닔!
INPUT_DIM = 5120  # ProtT5??1024李⑥썝?낅땲??

# =========================================================
# ?뱛 寃쎈줈 ?먮룞 ?ㅼ젙
# =========================================================
TRAIN_EMB_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_emb.npy')
TRAIN_IDS_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_ids.npy')
TEST_EMB_FILE  = os.path.join(EMBEDDING_DIR, 'testsuperset_emb.npy')
TEST_IDS_FILE  = os.path.join(EMBEDDING_DIR, 'testsuperset_ids.npy')

OUTPUT_DIR = './results'
TRAIN_TERMS = './data/raw/train/train_terms.tsv'

# ?섏씠?쇳뙆?쇰???(硫붾え由?遺議깆떆 BATCH_SIZE瑜?64濡?以꾩씠?몄슂)
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 10
SAVE_THRESHOLD = 0.01  
TOP_K = 150 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"?뵦 [{MODEL_NAME}] Two-Track Training Start... (Dim: {INPUT_DIM})")

# =========================================================
# ?㎚ ?곗씠?곗뀑 & ?좏떥由ы떚
# =========================================================
class UniversalDataset(Dataset):
    def __init__(self, emb_path, id_map, target_pids, labels_dict=None, num_classes=0):
        # mmap_mode='r'濡?硫붾え由??덉빟 (?뚯씪 ?꾩껜瑜?RAM???щ━吏 ?딆쓬)
        self.emb_matrix = np.load(emb_path, mmap_mode='r') 
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

# =========================================================
# ?룂 ?숈뒿 諛??덉륫 ?⑥닔
# =========================================================
def train_cycle(loss_name, criterion, output_path, train_loader, test_loader, num_classes):
    print(f"\n?? [{loss_name}] Training...")
    model = SimpleMLP(INPUT_DIM, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"{loss_name} Ep {epoch+1}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
    print(f"?뵰 [{loss_name}] Inference...")
    model.eval()
    with open(output_path, 'w') as f:
        with torch.no_grad():
            for x, pids in tqdm(test_loader, desc=f"{loss_name} Pred"):
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
    print(f"??Saved: {output_path}")

# =========================================================
# ?뵦 硫붿씤 濡쒖쭅
# =========================================================
if __name__ == "__main__":
    # 1. ?곗씠??以鍮?    print("   Preparing Data...")
    train_ids = clean_ids(np.load(TRAIN_IDS_FILE))
    train_id_map = {pid: i for i, pid in enumerate(train_ids)}
    test_ids = clean_ids(np.load(TEST_IDS_FILE))
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

    # Loader ?ㅼ젙
    train_ds = UniversalDataset(TRAIN_EMB_FILE, train_id_map, trn_ids, train_labels, len(top_terms))
    test_ds  = UniversalDataset(TEST_EMB_FILE,  test_id_map,  test_ids, num_classes=0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. BCE ?숈뒿 (Head)
    bce_out = f"{OUTPUT_DIR}/pred_{MODEL_NAME}_BCE.tsv"
    train_cycle("BCE", nn.BCEWithLogitsLoss(), bce_out, train_loader, test_loader, len(top_terms))

    # 3. ASL ?숈뒿 (Tail)
    asl_out = f"{OUTPUT_DIR}/pred_{MODEL_NAME}_ASL.tsv"
    train_cycle("ASL", AsymmetricLoss(), asl_out, train_loader, test_loader, len(top_terms))

    # 4. ?⑹껜 (硫붾え由??덉쟾 紐⑤뱶)
    print("\n?뵕 Merging BCE + ASL (Memory Safe)...")
    
    # [以묒슂] float32濡??쎌뼱??硫붾え由??덉빟
    df_bce = pd.read_csv(bce_out, sep='\t', names=['Id', 'Term', 'Score'], header=None, dtype={'Score': np.float32})
    df_asl = pd.read_csv(asl_out, sep='\t', names=['Id', 'Term', 'Score'], header=None, dtype={'Score': np.float32})
    
    # Outer Join
    merged = pd.merge(df_bce, df_asl, on=['Id', 'Term'], suffixes=('_bce', '_asl'), how='outer')
    
    # [以묒슂] ?먮낯 ?곗씠??利됱떆 ??젣?섏뿬 硫붾え由??뺣낫
    del df_bce, df_asl
    gc.collect() 
    
    # ?됯퇏 怨꾩궛
    merged.fillna(0.0, inplace=True)
    merged['Score'] = (merged['Score_bce'] + merged['Score_asl']) / 2
    
    # ???    final_out = f"{OUTPUT_DIR}/pred_{MODEL_NAME}_FINAL.tsv"
    print(f"?뮶 Saving to {final_out}...")
    merged['Score'] = merged['Score'].map(lambda x: '{:.5f}'.format(x))
    merged[['Id', 'Term', 'Score']].to_csv(final_out, sep='\t', index=False, header=False)
    
    print(f"?럦 理쒖쥌 ?꾨즺: {final_out}")

import os
# import gc
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from tqdm.auto import tqdm

# # =========================================================
# # ?숋툘 [?ㅼ젙] ESM-15B ?꾩슜 (李⑥썝 5120 二쇱쓽!)
# # =========================================================
# MODEL_NAME = "esm2_15b"
# # ?슚 ?대뜑紐??뺤씤 (15B ?대뜑)
# EMBEDDING_DIR = './data/embeddings/esm2_15B' 

# # ?슚 [?듭떖 ?섏젙] ESM-15B??李⑥썝??5120?낅땲?? (3B??2560)
# INPUT_DIM = 5120 

# # # =========================================================
# TRAIN_EMB_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_emb.npy')
# TRAIN_IDS_FILE = os.path.join(EMBEDDING_DIR, 'train_sequences_ids.npy')
# TEST_EMB_FILE  = os.path.join(EMBEDDING_DIR, 'testsuperset_emb.npy')
# TEST_IDS_FILE  = os.path.join(EMBEDDING_DIR, 'testsuperset_ids.npy')

# OUTPUT_DIR = './results'
# TRAIN_TERMS = './data/raw/train/train_terms.tsv'

# # ?뚮씪誘명꽣 (?덉쟾 紐⑤뱶)
# BATCH_SIZE = 32     # 李⑥썝??而ㅼ꽌 硫붾え由?留롮씠 癒뱀쓬 -> 32濡?異뺤냼
# LR = 1e-4           # 1e-3? ?덈Т ?쎈땲?? 1e-4濡???땄.
# EPOCHS = 10         
# SAVE_THRESHOLD = 0.01   
# TOP_K = 300       

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"?뵦 [{MODEL_NAME}] Training Start... (Dim: {INPUT_DIM})")

# # =========================================================
# # ?㎚ ?곗씠?곗뀑
# # =========================================================
# class UniversalDataset(Dataset):
#     def __init__(self, emb_path, id_map, target_pids, labels_dict=None, num_classes=0):
#         # mmap_mode濡?硫붾え由??덉빟 (15B???뚯씪????
#         self.emb_matrix = np.load(emb_path, mmap_mode='r') 
#         self.id_map = id_map
#         self.target_pids = target_pids
#         self.labels_dict = labels_dict
#         self.num_classes = num_classes
        
#     def __len__(self): return len(self.target_pids)
    
#     def __getitem__(self, idx):
#         pid = self.target_pids[idx]
#         if pid in self.id_map:
#             row_idx = self.id_map[pid]
#             emb = self.emb_matrix[row_idx]
#         else:
#             emb = np.zeros(INPUT_DIM, dtype=np.float32)
        
#         emb = torch.tensor(emb, dtype=torch.float32)
#         if self.labels_dict is not None:
#             label = np.zeros(self.num_classes, dtype=np.float32)
#             if pid in self.labels_dict:
#                 for t_idx in self.labels_dict[pid]: label[t_idx] = 1.0
#             return emb, torch.tensor(label)
#         else:
#             return emb, pid

# class SimpleMLP(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super().__init__()
#         # 5120李⑥썝??諛쏆븘以?異⑸텇???덉씠??#         self.layers = nn.Sequential(
#             nn.Linear(input_dim, 2048),
#             nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(0.4), # Dropout ?곹뼢
#             nn.Linear(2048, 1024),
#             nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
#             nn.Linear(1024, num_classes)
#         )
#     def forward(self, x): return self.layers(x)

# class AsymmetricLoss(nn.Module):
#     def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
#         super().__init__()
#         self.gamma_neg, self.gamma_pos, self.clip = gamma_neg, gamma_pos, clip
#     def forward(self, x, y):
#         x_sigmoid = torch.sigmoid(x)
#         xs_neg = (1 - x_sigmoid + self.clip).clamp(max=1)
#         loss = -1 * (self.gamma_pos * y * torch.log(x_sigmoid.clamp(min=1e-8)) + 
#                      self.gamma_neg * (1 - y) * torch.log(xs_neg.clamp(min=1e-8)) * (xs_neg**self.gamma_neg))
#         return loss.sum()

# def clean_ids(id_array):
#     id_list = []
#     flat_array = id_array.reshape(-1)
#     for pid in flat_array:
#         if isinstance(pid, bytes): pid = pid.decode('utf-8')
#         pid = str(pid).strip().replace('>', '')
#         if '|' in pid: parts = pid.split('|'); pid = parts[1] if len(parts) >= 2 else pid
#         id_list.append(pid)
#     return id_list

# # =========================================================
# # ?룂 硫붿씤 ?ㅽ뻾
# # =========================================================
# if __name__ == "__main__":
#     if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

#     # 1. ?곗씠??以鍮?#     print("   Preparing Data...")
#     try:
#         train_ids = clean_ids(np.load(TRAIN_IDS_FILE))
#         test_ids = clean_ids(np.load(TEST_IDS_FILE))
#     except FileNotFoundError:
#         print("?슚 ?뚯씪 寃쎈줈 ?뺤씤 ?꾩슂! 15B ?대뜑媛 留욌굹??"); exit()

#     train_id_map = {pid: i for i, pid in enumerate(train_ids)}
#     test_id_map = {pid: i for i, pid in enumerate(test_ids)}

#     train_terms = pd.read_csv(TRAIN_TERMS, sep='\t')
#     term_counts = train_terms['term'].value_counts()
#     top_terms = term_counts[term_counts >= 10].index.tolist()
    
#     term2idx = {t: i for i, t in enumerate(top_terms)}
#     idx2term = {i: t for i, t in enumerate(top_terms)}
    
#     train_labels = {}
#     for pid, term in zip(train_terms['EntryID'], train_terms['term']):
#         pid = str(pid).strip()
#         if term in term2idx and pid in train_id_map:
#             if pid not in train_labels: train_labels[pid] = []
#             train_labels[pid].append(term2idx[term])

#     valid_pids = list(train_labels.keys())
#     trn_ids, _ = train_test_split(valid_pids, test_size=0.1, random_state=42)

#     train_ds = UniversalDataset(TRAIN_EMB_FILE, train_id_map, trn_ids, train_labels, len(top_terms))
#     test_ds  = UniversalDataset(TEST_EMB_FILE,  test_id_map,  test_ids, num_classes=0)
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#     test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

#     # ?뵦 ASL ?숈뒿 (LR ??땄)
#     print(f"\n?? [ESM-15B] Safe Training Start...")
#     model = SimpleMLP(INPUT_DIM, len(top_terms)).to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LR) # 1e-4
#     criterion = AsymmetricLoss()

#     for epoch in range(EPOCHS):
#         model.train()
#         total_loss = 0
#         for x, y in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
#             x, y = x.to(device), y.to(device)
#             optimizer.zero_grad()
#             loss = criterion(model(x), y)
#             loss.backward()
            
#             # [?덉쟾?μ튂] Gradient Clipping (??＜ 諛⑹?)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")

#     # 異붾줎
#     output_path = f"{OUTPUT_DIR}/pred_{MODEL_NAME}_FINAL.tsv"
#     print(f"?뵰 Inference -> {output_path}")
#     model.eval()
#     with open(output_path, 'w') as f:
#         with torch.no_grad():
#             for x, pids in tqdm(test_loader, desc="Predicting"):
#                 x = x.to(device)
#                 probs = torch.sigmoid(model(x)).cpu().numpy()
#                 for i, pid in enumerate(pids):
#                     p = probs[i]
#                     # ?먯닔媛 0.999留??섏삤?붿? 泥댄겕
#                     if np.mean(p) > 0.9: 
#                         continue # 鍮꾩젙???곗씠??嫄대꼫? (?덉쟾?μ튂)

#                     indices = np.where(p >= SAVE_THRESHOLD)[0]
#                     scores = p[indices]
#                     if len(scores) > TOP_K:
#                         top_idx = np.argsort(scores)[-TOP_K:]
#                         indices = indices[top_idx]
#                         scores = scores[top_idx]
#                     for idx, score in zip(indices, scores):
#                         f.write(f"{pid}\t{idx2term[idx]}\t{score:.5f}\n")
    
#     print(f"?럦 ESM-15B ?뺤긽???꾨즺: {output_path}")
