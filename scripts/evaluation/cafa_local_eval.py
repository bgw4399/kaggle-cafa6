# cafa_local_eval.py
import argparse
import os
import shutil
import glob
import subprocess
import pandas as pd
import tempfile

def ensure_pred_dir(pred_path: str):
    """cafaeval은 'prediction_folder'를 받으므로, 파일 1개면 임시 폴더를 만들어 넣어줌."""
    if os.path.isdir(pred_path):
        return pred_path, None
    tmp = tempfile.mkdtemp(prefix="preds_")
    dst = os.path.join(tmp, os.path.basename(pred_path))
    shutil.copy(pred_path, dst)
    return tmp, tmp

def pick_best_wf_file(out_dir: str):
    # Weighted-F best 파일을 우선 찾고, 없으면 best 아무거나
    cands = sorted(glob.glob(os.path.join(out_dir, "evaluation_best_*wf*.tsv")))
    if cands:
        return cands[0]
    cands = sorted(glob.glob(os.path.join(out_dir, "evaluation_best_*.tsv")))
    if not cands:
        raise FileNotFoundError(f"No evaluation_best_*.tsv in {out_dir}")
    return cands[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obo", required=True, help="GO .obo file path")
    ap.add_argument("--gt", required=True, help="Ground truth TSV (Id<TAB>Term)")
    ap.add_argument("--pred", required=True, help="Prediction file or folder (Id<TAB>Term<TAB>Score)")
    ap.add_argument("--ia", default=None, help="IA.tsv (optional, enables weighted metrics)")
    ap.add_argument("--out_dir", default="./cafa_eval_out")
    ap.add_argument("--prop", default="max", choices=["max", "fill"])
    ap.add_argument("--no_orphans", action="store_true")
    ap.add_argument("--norm", default="cafa", choices=["cafa", "pred", "gt"])
    ap.add_argument("--th_step", type=float, default=0.01)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    pred_dir, tmpdir = ensure_pred_dir(args.pred)

    os.makedirs(args.out_dir, exist_ok=True)

    cmd = [
        "cafaeval",
        args.obo,
        pred_dir,
        args.gt,
        "-out_dir", args.out_dir,
        "-prop", args.prop,
        "-norm", args.norm,
        "-th_step", str(args.th_step),
        "-threads", str(args.threads),
    ]
    if args.ia:
        cmd += ["-ia", args.ia]
    if args.no_orphans:
        cmd += ["-no_orphans"]

    print("🚀 Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    best_file = pick_best_wf_file(args.out_dir)
    df = pd.read_csv(best_file, sep="\t")

    # wf 컬럼명은 환경/버전에 따라 다를 수 있어서 안전하게 찾음
    wf_col = None
    for c in df.columns:
        if c.lower() in ("wf", "weighted_f", "weighted_f_measure"):
            wf_col = c
            break
    if wf_col is None:
        # 그래도 없으면 wf로 시작하는 컬럼 탐색
        for c in df.columns:
            if c.lower().startswith("wf"):
                wf_col = c
                break

    print(f"\n✅ Best file: {best_file}")
    if wf_col is None:
        print("⚠️ Can't find weighted-F column. Columns:", list(df.columns))
        print(df.head(10))
    else:
        # namespace 별 best row 출력
        if "ns" in df.columns:
            for ns, sub in df.groupby("ns"):
                row = sub.iloc[0]
                tau = row["tau"] if "tau" in row else None
                print(f"- {ns}: {wf_col}={row[wf_col]:.6f}  tau={tau}")
        else:
            print(df[[wf_col]].head(10))

    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
