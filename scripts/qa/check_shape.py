import numpy as np
try:
    arr = np.load('./data/embeddings/protT5_xl/train_sequences_emb.npy', mmap_mode='r')
    print(arr.shape)
except Exception as e:
    print(e)

