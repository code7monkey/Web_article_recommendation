# Jump AI(py) 2025 : 뉴스 추천 경진대회 샘플 프로젝트

**NewsRec 2025 : 제3회 뉴스 추천 경진대회 예시 코드**  
**참고용 베이스라인 – 협업 필터링 & 콘텐츠 기반 추천**

---

본 저장소는 간단한 뉴스 추천 시스템을 구현한 예시 프로젝트입니다.  
사용자의 조회 로그와 기사 정보를 이용하여 **SMILES 입력 기반 pIC50 회귀** 대신, **사용자–기사 행렬을 활용한 협업 필터링**과  
**기사 내용의 TF‑IDF 임베딩을 활용한 콘텐츠 기반 추천**을 결합하는 방식으로 추천 점수를 계산합니다.  
학습 코드와 추론 코드를 분리하고, **YAML 설정 파일**을 통해 실험을 제어할 수 있도록 구성되어 있습니다.

---

## 🎯 Project Goals

* **사용자 조회 기록 기반 협업 필터링** – 사용자들의 기사 조회 행태를 행렬로 표현하고, 코사인 유사도를 통해 비슷한 사용자를 찾습니다.
* **기사 내용 기반 콘텐츠 추천** – 기사의 `Content` 컬럼을 TF‑IDF로 임베딩하여 기사 간 유사도를 계산합니다.
* **Hybrid 추천 점수** – 협업 필터링 점수와 콘텐츠 기반 점수를 결합하여 최종 추천 점수를 만듭니다.
* **Scaffold 기반 K‑Fold가 아닌 사용자별 추천** – 사용자가 작성한 기사와 반복적으로 읽은 기사를 적절히 추천 목록에 포함합니다.
* **YAML 기반 설정 관리** – 학습 스크립트(`train.py`)와 추론 스크립트(`inference.py`)에서 공통 설정을 `configs/*.yaml` 파일로 분리하여 코드 수정 없이 실험을 조정할 수 있습니다.

---

## 📁 Project Structure

```
newsrec_project2/
├── src/                    # 핵심 로직 (import용)
│   ├── __init__.py
│   ├── dataset.py          # 데이터 로딩 및 전처리 함수
│   ├── model.py            # 협업 필터링 및 콘텐츠 기반 점수 계산
│   ├── trainer.py          # 추천 점수 결합 및 추천 목록 생성
│   ├── losses.py           # (확장용) 커스텀 손실 함수 자리
│   └── utils.py            # 공용 함수 (seed 고정 등)
│
├── train.py                # 학습 실행 스크립트 (YAML 설정 읽기)
├── inference.py            # 추론 / 제출 파일 생성 스크립트
│
├── configs/                # 설정 파일 (코드 수정 없이 실험 제어)
│   ├── train.yaml          # 데이터 경로, 가중치 등 학습 설정
│   └── submit.yaml         # 추론 및 제출 설정
│
├── assets/                 # 중간 산출물 (예: 훈련된 모델, 전처리 정보)
│   ├── combined_scores.npy  # Hybrid 추천 점수 행렬 (train.py에서 생성)
│   └── user_data.pkl       # 사용자별 작성 기사/조회 정보 (추론용)
│
├── data/                   # 입력 데이터 (샘플 포함)
│   ├── view_log.csv        # 사용자의 기사 조회 로그
│   ├── article_info.csv    # 기사 내용 (Content 컬럼 포함)
│   └── sample_submission.csv # 제출 포맷 예시
│
├── requirements.txt        # 실행 환경 고정
├── .gitignore              # Git이 무시할 파일 패턴
├── .gitattributes          # Git 설정 (예: LFS)
└── README.md
```

---

## 🛠 Environment Setup

Python 3.9 이상에서 동작하도록 작성되었습니다. 프로젝트 루트에서 다음 명령으로 의존성을 설치합니다.

```bash
pip install -r requirements.txt
```

GPU가 필요한 딥러닝 모델이 없으므로 CPU 환경에서도 실행 가능합니다.

---

## 🚀 Usage

학습과 추론은 분리된 스크립트로 제공됩니다. 각각의 스크립트는 `configs/` 아래 YAML 설정 파일을 읽어 동작하므로, 하이퍼파라미터나 데이터 경로를 수정할 때 스크립트를 건드릴 필요가 없습니다.

### Train

```bash
python train.py --config configs/train.yaml
```

`train.py`는 다음 작업을 수행합니다:

1. `dataset.py`를 통해 `view_log.csv`와 `article_info.csv`를 로딩하고 사용자–기사 행렬을 생성합니다.
2. `model.py`의 함수를 이용하여 협업 필터링 점수와 콘텐츠 기반 점수를 계산합니다.
3. `trainer.py`를 호출하여 두 점수를 합성하고 추천 점수를 저장합니다.
4. 필요에 따라 중간 결과를 `assets/` 디렉터리에 저장하여 추론 단계에서 재사용합니다.

### Inference

```bash
python inference.py --config configs/submit.yaml
```

`inference.py`는 학습 단계에서 저장한 추천 점수를 불러와 `sample_submission.csv` 포맷에 맞는 제출 파일을 생성합니다.  
사용자가 작성한 기사나 반복 읽은 기사를 우선순위에 포함하는 로직 또한 `trainer.py`에 구현되어 있습니다.

---

## 📜 Notes

* `losses.py`에는 현재 사용되지 않는 커스텀 손실 함수를 구현할 수 있는 자리가 마련돼 있습니다. 협업 필터링 모델을 학습형 모델로 확장할 때 활용해 보세요.
* `.gitignore`에는 `data/`, `assets/`, `.ipynb_checkpoints/` 등의 폴더와 대용량 파일이 포함되어 있어 Git 저장소 용량을 줄입니다.
* `requirements.txt`는 예시로 `pandas`, `scikit-learn` 등의 주요 라이브러리를 버전과 함께 명시합니다. 실제 환경에 맞게 수정하세요.

이 예제 프로젝트는 간단한 하이브리드 추천 시스템의 구조를 보여 주기 위한 것입니다. 실제 대회에 참가할 때는 더 정교한 모델링과 데이터 전처리가 필요할 수 있습니다.
