import numpy as np
import os

BASE_DIR = "./data/embeddings"

def check_dim(name, path):
    try:
        if path.endswith('.npy'):
            data = np.load(path, mmap_mode='r')
            print(f"??{name}: {data.shape}")
        elif path.endswith('.npz'):
            data = np.load(path)
            # usually 'arr_0' or similar
            keys = list(data.keys())
            print(f"??{name} (keys: {keys}): {data[keys[0]].shape}")
    except Exception as e:
        print(f"??{name}: {e}")

check_dim("ProtT5", os.path.join(BASE_DIR, "protT5_xl/train_sequences_emb.npy"))
check_dim("Ankh", os.path.join(BASE_DIR, "ankh_large/train_sequences_emb.npy"))
check_dim("PPI", os.path.join(BASE_DIR, "ppi/ppi_features.npz"))


