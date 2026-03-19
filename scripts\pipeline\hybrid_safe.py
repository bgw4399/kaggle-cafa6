import pandas as pd
import numpy as np
import os
import shutil
import gc
from tqdm import tqdm

# =========================================================
# ⚙️ [설정] 메모리 터짐 방지 Dynamic Hybrid
# =========================================================
# 1. 입력 파일 (경로 확인 필수!)
# Deep Learning 결과 (순정 ESM 파일 추천)
FILE_DL = './results/submission_step1_aggregated.tsv'
# Diamond 결과 (Filtered 버전 필수)
FILE_DIA = './results/submission_diamond_taxon_filtered.tsv'

# 2. 결과 파일
OUTPUT_FILE = './results/submission_Dynamic_Hybrid_Safe.tsv'

# 3. 안전 장치 (20개 분할)
NUM_BUCKETS = 20
TEMP_DIR = './temp_hybrid_buckets'
# =========================================================

def get_bucket_id(pid):
    if isinstance(pid, bytes): pid = pid.decode('utf-8')
    return hash(pid) % NUM_BUCKETS

def split_to_buckets(filepath, prefix):
    print(f"🔨 Splitting {prefix} into {NUM_BUCKETS} buckets...")
    if not os.path.exists(filepath):
        print(f"❌ 파일 없음: {filepath}")
        return False
    
    # 파일 핸들 미리 열기 (속도 최적화)
    handles = {}
    for i in range(NUM_BUCKETS):
        os.makedirs(os.path.join(TEMP_DIR, str(i)), exist_ok=True)
        path = os.path.join(TEMP_DIR, str(i), f"{prefix}.tsv")
        handles[i] = open(path, 'w')
        
    try:
        with open(filepath, 'r') as f:
            for line in tqdm(f, desc=f"Reading {prefix}"):
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                
                # PID를 기준으로 해싱 -> 같은 단백질은 같은 버킷으로 감
                pid = parts[0]
                b_id = get_bucket_id(pid)
                handles[b_id].write(line)
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        for h in handles.values(): h.close()
    return True

def process_buckets():
    print(f"\n🚀 Processing Buckets (Dynamic Hybrid Logic)...")
    
    with open(OUTPUT_FILE, 'w') as fout: pass # 초기화
    
    for b_idx in tqdm(range(NUM_BUCKETS), desc="Merging"):
        bucket_dir = os.path.join(TEMP_DIR, str(b_idx))
        
        # 1. DL 로드
        dl_map = {}
        path_dl = os.path.join(bucket_dir, "DL.tsv")
        if os.path.exists(path_dl):
            with open(path_dl, 'r') as f:
                for line in f:
                    p, t, s = line.strip().split('\t')
                    dl_map[(p, t)] = float(s)

        # 2. Diamond 로드
        dia_map = {}
        path_dia = os.path.join(bucket_dir, "DIA.tsv")
        if os.path.exists(path_dia):
            with open(path_dia, 'r') as f:
                for line in f:
                    p, t, s = line.strip().split('\t')
                    dia_map[(p, t)] = float(s)
        
        # 3. 합집합 키 생성
        all_keys = set(dl_map.keys()) | set(dia_map.keys())
        
        lines = []
        for key in all_keys:
            s_dl = dl_map.get(key, 0.0)
            s_dia = dia_map.get(key, 0.0)
            
            final_score = 0.0
            
            # 🔥 [Dynamic Hybrid 로직 적용]
            # 1. Diamond 확신 (1.0에 가까움) -> 무조건 믿음 (검색 승리)
            if s_dia >= 0.99:
                final_score = 1.0
            
            # 2. Diamond 중박 (0.5 ~ 0.99) -> DL과 비교해서 큰 것
            elif s_dia > 0.5:
                final_score = max(s_dl, s_dia)
            
            # 3. Diamond 모름 -> DL 점수 믿음
            else:
                final_score = s_dl
            
            # 저장 (0.001 미만 삭제)
            if final_score > 0.001:
                lines.append(f"{key[0]}\t{key[1]}\t{final_score:.5f}\n")
        
        # 4. 쓰기
        with open(OUTPUT_FILE, 'a') as fout:
            fout.writelines(lines)
            
        # 메모리 해제 (중요!)
        del dl_map, dia_map, all_keys, lines
        gc.collect()

if __name__ == "__main__":
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    print("🔋 Safe Hybrid Merge Start...")
    
    # 1. 쪼개기 (RAM 절약)
    split_to_buckets(FILE_DL, "DL")
    split_to_buckets(FILE_DIA, "DIA")
    
    # 2. 합치기
    process_buckets()
    
    # 3. 청소
    shutil.rmtree(TEMP_DIR)
    print(f"🎉 완료! 결과 파일: {OUTPUT_FILE}")
    
    print("\n👉 [Next Step]")
    print("1. 이 파일이 생성되면, 아까 말씀하신 'UniProtKB Diamond' 결과가 나올 때까지 기다리세요.")
    print("2. 그 다음, 이 파일과 UniProt 결과를 'Max Merge'하면 0.46 도전을 위한 준비가 끝납니다.")