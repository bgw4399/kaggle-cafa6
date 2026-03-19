# CAFA6 Experiments

CAFA6 단백질 기능 예측 실험을 정리한 저장소입니다. 코드, 노트북, 참조 파일, 로컬 전용 데이터/산출물을 분리해서 GitHub에 올리기 쉽게 정리했습니다.

## 폴더 구조

```text
.
|-- scripts/
|   |-- analysis/
|   |-- data_prep/
|   |-- evaluation/
|   |-- experimental/
|   |-- pipeline/
|   |-- qa/
|   `-- training/
|-- notebooks/
|-- data/
|   |-- reference/      # GitHub에 포함 가능한 참조 파일
|   |-- raw/            # 원본 데이터 (git ignore)
|   |-- embeddings/     # 임베딩 파일 (git ignore)
|   |-- derived/        # 파생 산출물 (git ignore)
|   `-- metadata/       # 학습 메타데이터 (git ignore)
|-- results/            # 예측/실험 결과물 (git ignore)
`-- artifacts/          # 검색 DB, 임시 로그, sqlite 조각 등 (git ignore)
```

## 작업 규칙

- 모든 스크립트는 저장소 루트에서 실행하는 것을 기준으로 정리했습니다.
- CAFA 원본 입력은 `data/raw/train`, `data/raw/test` 아래에 둡니다.
- 임베딩은 `data/embeddings` 아래에 둡니다.
- 생성 결과는 `results/` 아래에 쌓이도록 맞췄습니다.

예시:

```bash
python scripts/training/train.py
python scripts/data_prep/create_val_split.py
python scripts/pipeline/predict_stacking.py
```

## 설치

```bash
pip install -r requirements.txt
```

## GitHub에 포함되는 것

- `scripts/`
- `notebooks/`
- `data/reference/`
- `README.md`
- `requirements.txt`

## GitHub에서 제외되는 것

- 원본 데이터
- 임베딩
- 결과 파일
- 검색용 DB/임시 파일
- 파생 메타데이터
