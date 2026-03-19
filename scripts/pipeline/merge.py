import pandas as pd

import os

import gc



# =========================================================

# ?뱛 [?ㅼ젙] ?뚯씪 寃쎈줈 ?뺤씤

# =========================================================

# 1. 諛⑷툑 0.256???섏삩 ESM-3B ?뚯씪

DL_FILE = './results/submission_step1_aggregated.tsv.tsv'



# 2. Diamond ?뚯씪 (議깅낫)

DIAMOND_FILE = './results/submission_diamond_fixed.tsv'



# 3. GAF ?뺣떟吏 (移섑듃??1: 臾댁“嫄??뺣떟 泥섎━)

GAF_POS_FILE = './gaf_positive_preds.tsv'



# 4. GAF ?ㅻ떟吏 (移섑듃??2: 臾댁“嫄???젣 泥섎━)

GAF_NEG_FILE = './gaf_negative_preds.tsv'



# 5. 理쒖쥌 ?쒖텧 ?뚯씪紐?
OUTPUT_FILE = './results/submission_final_sota.tsv'



# ?숋툘 ?뚮씪誘명꽣

DIAMOND_THRESHOLD = 0.75  # 0.75 ?댁긽留?誘우쓬

# =========================================================



def load_data(path, names=['Id', 'Term', 'Score']):

    if not os.path.exists(path):

        print(f"?좑툘 ?뚯씪 ?놁쓬: {path} (泥댄겕 ?꾩슂!)")

        return None

    print(f"?뱛 Loading {path}...")

    return pd.read_csv(path, sep='\t', names=names, header=None)



print("?? Final Ensemble ?쒖옉 (ESM + Diamond + GAF)...")



# ---------------------------------------------------------

# 1. DL & Diamond ?⑹껜 (Max ?꾨왂)

# ---------------------------------------------------------

df_dl = load_data(DL_FILE)

df_dia = load_data(DIAMOND_FILE)



# Diamond 而룹삤??
if df_dia is not None:

    print(f"   ?뭿 Diamond ?꾪꽣留?(Score >= {DIAMOND_THRESHOLD})")

    df_dia = df_dia[df_dia['Score'] >= DIAMOND_THRESHOLD].copy()



print("??1李??⑹껜: Max(DL, Diamond)...")

# DL怨?Diamond瑜??⑹튂怨? 媛숈? (Id, Term)????????믪? ?먯닔瑜??좏깮

combined = pd.concat([df_dl, df_dia])

final_df = combined.groupby(['Id', 'Term'], as_index=False)['Score'].max()



# 硫붾え由??뺣━

del df_dl, df_dia, combined

gc.collect()



# ---------------------------------------------------------

# 2. GAF Positive ?곸슜 (臾댁“嫄?1.0 留뚮뱾湲?

# ---------------------------------------------------------

df_pos = load_data(GAF_POS_FILE)

if df_pos is not None:

    print(f"??GAF Positive ?곸슜 ({len(df_pos):,}媛?...")

    # GAF Positive ?뚯씪???먯닔???대? 1.0??寃껋씠誘濡? concat ??max ?섎㈃ 1.0?쇰줈 ??뼱?⑥쭚

    final_df = pd.concat([final_df, df_pos])

    final_df = final_df.groupby(['Id', 'Term'], as_index=False)['Score'].max()

    del df_pos

    gc.collect()



# ---------------------------------------------------------

# 3. GAF Negative ?곸슜 (臾댁“嫄???젣?섍린)

# ---------------------------------------------------------

df_neg = load_data(GAF_NEG_FILE)

if df_neg is not None:

    print(f"?㏏ GAF Negative ?곸슜 (?ㅻ떟 ??젣)...")

    

    # ??젣 ?띾룄瑜??꾪빐 Set?쇰줈 蹂??
    neg_set = set(zip(df_neg['Id'], df_neg['Term']))

    

    # (Id, Term)???몃뜳?ㅻ줈 留뚮뱾?댁꽌 ??젣

    final_df_idx = final_df.set_index(['Id', 'Term'])

    

    # ??젣?????李얘린

    drop_indices = [x for x in final_df_idx.index if x in neg_set]

    

    if drop_indices:

        print(f"   ?슟 {len(drop_indices):,}媛쒖쓽 ?뺤떎???ㅻ떟???쒓굅?⑸땲??")

        final_df_idx = final_df_idx.drop(index=drop_indices)

        final_df = final_df_idx.reset_index()

    else:

        print("   ??寃뱀튂???ㅻ떟???놁뒿?덈떎.")

        

    del df_neg, neg_set

    gc.collect()



# ---------------------------------------------------------

# 4. ???
# ---------------------------------------------------------

print(f"?뮶 Saving to {OUTPUT_FILE}...")

final_df['Score'] = final_df['Score'].map(lambda x: '{:.5f}'.format(x))

final_df.to_csv(OUTPUT_FILE, sep='\t', index=False, header=False)



print("\n?럦 紐⑤뱺 ?묒뾽 ?꾨즺!")

print(f"?몛 ?쒖텧 ?뚯씪: {OUTPUT_FILE}")

print("?몛 ???뚯씪??0.256(DL) + 0.35(Diamond) + GAF 吏?앹쓣 紐⑤몢 ?댁? 理쒖쥌蹂몄엯?덈떎.")
