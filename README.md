# Web News Recommendation Challenge – Sample Project

This repository provides a **sample hybrid recommendation pipeline** for a web news recommendation challenge.  
It combines **content-based recommendation using TF-IDF embeddings** with **collaborative filtering based on user view logs** to compute final recommendation scores.

Training and inference are separated into different scripts, and all experiments are controlled through **YAML configuration files**, allowing easy tuning without modifying source code.

---

## 🎯 Project Goals

- **Collaborative filtering based on user viewing history**  
  Represent user–article interactions as a matrix and identify similar users using cosine similarity.

- **Content-based recommendation using article text**  
  Embed article contents (the `Content` column) with TF-IDF and compute article-to-article similarity.

- **Hybrid recommendation scoring**  
  Combine collaborative filtering scores and content-based similarity scores into a final recommendation score.

- **User-centric recommendation logic**  
  Recommend articles written by the user or frequently read articles when appropriate, instead of using scaffold-based K-Fold logic.

- **YAML-based configuration management**  
  Control training and inference behavior through `configs/*.yaml` files without changing code.

---

## 📁 Project Structure

    newsrec_project2/
    ├── src/                    # Core logic (importable modules)
    │   ├── __init__.py
    │   ├── dataset.py          # Data loading and preprocessing
    │   ├── model.py            # Collaborative & content-based scoring
    │   ├── trainer.py          # Score combination and recommendation logic
    │   ├── losses.py           # (Optional) placeholder for custom losses
    │   └── utils.py            # Common utilities (e.g., seed fixing)
    │
    ├── train.py                # Training script (reads YAML config)
    ├── inference.py            # Inference & submission generation script
    │
    ├── configs/                # Configuration files
    │   ├── train.yaml          # Training settings (paths, weights, etc.)
    │   └── submit.yaml         # Inference & submission settings
    │
    ├── assets/                 # Intermediate artifacts
    │   ├── combined_scores.npy # Hybrid recommendation score matrix
    │   └── user_data.pkl       # User-specific metadata for inference
    │
    ├── data/                   # Input data (sample included)
    │   ├── view_log.csv        # User article view logs
    │   ├── article_info.csv    # Article metadata (includes Content column)
    │   └── sample_submission.csv # Example submission format
    │
    ├── requirements.txt        # Fixed execution environment
    ├── .gitignore              # Git ignore patterns
    ├── .gitattributes          # Git settings (e.g., LFS)
    └── README.md

---

## 🛠 Environment Setup

The project is designed to run on **Python 3.9 or later**.  
Install dependencies from the project root:

    pip install -r requirements.txt

No deep learning models are used, so the pipeline can be executed entirely on **CPU environments**.

---

## 🚀 Usage

Training and inference are provided as separate scripts.  
Both scripts load settings from YAML files under `configs/`, enabling easy experiment control.

---

### Training

    python train.py --config configs/train.yaml

The training process performs the following steps:

1. Load `view_log.csv` and `article_info.csv` using `dataset.py` and construct a user–article interaction matrix.
2. Compute collaborative filtering scores and content-based similarity scores using functions in `model.py`.
3. Combine the scores via `trainer.py` and store the final recommendation score matrix.
4. Save intermediate artifacts to the `assets/` directory for reuse during inference.

---

### Inference

    python inference.py --config configs/submit.yaml

The inference script:

- Loads precomputed recommendation scores from the training stage
- Applies user-specific logic (e.g., prioritizing authored or frequently read articles)
- Generates a submission file following the `sample_submission.csv` format

---

## 📜 Notes

- `losses.py` is currently unused and serves as a placeholder for future extensions, such as learning-based collaborative filtering models.
- `.gitignore` excludes directories such as `data/`, `assets/`, and `.ipynb_checkpoints/` to keep the repository lightweight.
- `requirements.txt` lists core libraries such as `pandas` and `scikit-learn` with fixed versions. Adjust as needed for your environment.

This sample project demonstrates the **basic structure of a hybrid recommendation system**.  
For real competition use, more advanced modeling and feature engineering may be required.
