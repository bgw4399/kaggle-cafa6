# make_val_split_and_gt.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_terms", required=True, help="./data/raw/train/train_terms.tsv")
    ap.add_argument("--out_dir", default="./local_val")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.train_terms, sep="\t")
    df["EntryID"] = df["EntryID"].astype(str).str.strip()
    df["term"] = df["term"].astype(str).str.strip()

    proteins = df["EntryID"].unique()
    trn, val = train_test_split(
        proteins, test_size=args.val_ratio, random_state=args.seed
    )
    val_set = set(val)

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    # val ids
    with open(f"{args.out_dir}/val_ids.txt", "w") as f:
        for pid in val:
            f.write(f"{pid}\n")

    # val ground truth: (Id, Term)
    gt = df[df["EntryID"].isin(val_set)][["EntryID", "term"]]
    gt.to_csv(f"{args.out_dir}/val_ground_truth.tsv", sep="\t", index=False, header=False)

    print("??Done")
    print(f"- val ids: {args.out_dir}/val_ids.txt  ({len(val):,} proteins)")
    print(f"- val gt : {args.out_dir}/val_ground_truth.tsv  ({len(gt):,} rows)")

if __name__ == "__main__":
    main()


