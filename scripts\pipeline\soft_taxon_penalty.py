# soft_taxon_penalty.py
import argparse
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def clean_id_str(pid):
    if isinstance(pid, bytes):
        pid = pid.decode("utf-8")
    pid = str(pid).strip().replace(">", "")
    if "|" in pid:
        parts = pid.split("|")
        if len(parts) >= 2:
            pid = parts[1]
    return pid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_in", default="./results/final_submission/submission_Final_Hybrid.tsv")
    ap.add_argument("--pred_out",default="./results/final_submission/submission_Final_Hybrid_Taxon_Filtered.tsv")

    ap.add_argument("--train_terms", default="./data/raw/train/train_terms.tsv")
    ap.add_argument("--train_ids_npy", default="./data/embeddings/esm2_15B/train_sequences_ids.npy")
    ap.add_argument("--train_taxon_npy", default="./data/embeddings/taxonomy/train_species_idx.npy")

    ap.add_argument("--test_ids_npy", default="./data/embeddings/testsuperset_ids.npy")
    ap.add_argument("--test_taxon_npy", default="./data/embeddings/taxonomy/test_species_idx.npy")

    ap.add_argument("--penalty", type=float, default=0.3, help="?녿뜕 term ?먯닔??怨깊븷 媛먯뇿")
    ap.add_argument("--restore_threshold", type=float, default=0.85,
                    help="???먯닔 ?댁긽?대㈃ species???놁뼱???먯젏???좎?")
    args = ap.parse_args()

    # train species -> known terms
    raw_train_ids = np.load(args.train_ids_npy)
    train_ids = [clean_id_str(x) for x in raw_train_ids.reshape(-1)]
    train_species = np.load(args.train_taxon_npy).reshape(-1)
    id2sp = dict(zip(train_ids, train_species))

    sp2terms = {}
    train_df = pd.read_csv(args.train_terms, sep="\t")
    for pid, term in tqdm(zip(train_df["EntryID"], train_df["term"]), total=len(train_df), desc="build sp2terms"):
        pid = str(pid).strip()
        if pid not in id2sp:
            continue
        sp = id2sp[pid]
        sp2terms.setdefault(sp, set()).add(str(term).strip())

    # test pid -> species
    raw_test_ids = np.load(args.test_ids_npy)
    test_pids = [clean_id_str(x) for x in raw_test_ids.reshape(-1)]
    test_species = np.load(args.test_taxon_npy).reshape(-1)
    test_id2sp = dict(zip(test_pids, test_species))

    with open(args.pred_in, "r") as f_in, open(args.pred_out, "w") as f_out:
        for line in tqdm(f_in, desc="apply penalty"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pid = clean_id_str(parts[0])
            if pid not in test_id2sp:
                continue
            term = parts[1].strip()
            try:
                score = float(parts[2])
            except:
                continue

            sp = test_id2sp[pid]
            if sp in sp2terms and term not in sp2terms[sp]:
                # ?녿뜕 term: ??젣 ???媛먯뇿 (?? 留ㅼ슦 ?뺤떊?대㈃ ?좎?)
                if score < args.restore_threshold:
                    score *= args.penalty

            if score > 0:
                f_out.write(f"{pid}\t{term}\t{score:.5f}\n")

    print(f"??saved: {args.pred_out}")

if __name__ == "__main__":
    main()

