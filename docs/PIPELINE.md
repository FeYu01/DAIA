DAIA — Project Pipeline (High-level)

This document summarizes the end-to-end pipeline for the DAIA project (Detect AI-generated Images for Auto-insurance). It's written so you can copy it directly and paste it into other documentation or a report.

---

## 1) Data (acquire & store)

- Source types
  - Real images: public datasets / Kaggle extracts.
  - AI-generated images: outputs from image generators (e.g., Gemini, Midjourney).

- Storage location (current setup)
  - Active dataset directory (outside repo): `/home/codespace/datasets/DAIA`
    - `/home/codespace/datasets/DAIA/real`
    - `/home/codespace/datasets/DAIA/ai_generated`
  - Repository policy: large images are ignored by Git via `.gitignore`. A small placeholder (`data/real/.gitkeep`) is kept so folder structure remains in the repo.

- Backup & versioning
  - Keep an external backup (cloud or local) if you need sharing or long-term storage.
  - For dataset versioning or team sharing, use DVC with a cloud remote or use Git LFS for simpler cases.

## 2) Data hygiene & preprocessing

- Watermark handling
  - Some AI generators add watermarks (e.g., Gemini). Use the provided scripts to detect and crop these safely:
    - `scripts/crop_gemini_only.py` — backs up originals then crops a tuned bottom strip (e.g., 80 px) for Gemini images.
    - `scripts/crop_watermark_batch.py` and `scripts/test_watermark_crop.py` — for batch runs and tuning.

- Loading & transforms
  - Central loader: `src/data_loader.py`.
  - Images are resized to the model input size using Albumentations `A.Resize(image_size, image_size)` (default 224x224 from `config.yaml`).
  - Training augmentations include random flip, rotation, color jitter, Gaussian noise, and blur via `get_transforms(config, is_training=True)`.
  - Splits are reproducible using the seed in `config.yaml` (train/val/test ratios are configurable there).

## 3) Configuration & repo wiring

- Central config
  - `config.yaml` contains all key parameters (data paths, image size, batch size, seed, training hyperparameters, checkpoint paths).
  - The code loads `config.yaml` (via `src/utils.py` loader) so changes are centralized.

- Git hygiene
  - `.gitignore` excludes large files and the `data/` image directories.
  - Actual dataset files were moved to `/home/codespace/datasets/DAIA` and removed from the Git index so the repo stays lightweight.

## 4) Training pipeline (high-level)

- Entry point
  - `src/train.py` orchestrates training: loads config, builds datasets and dataloaders, constructs the model and optimizer, runs epochs with validation, and saves checkpoints.

- Model architecture
  - Vision Transformer (ViT) backbone (e.g., `google/vit-base-patch16-224-in21k`) with a small binary classification head.
  - Transfer learning: freeze/unfreeze strategy can be applied; default is to fine-tune the pre-trained encoder and train a small head.

- Loss and metrics
  - Loss: Cross-entropy (or BCEWithLogits depending on head).
  - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC. Validation metrics are logged per epoch.

- Checkpointing
  - Save weights (`state_dict`), optimizer state, epoch, and a snapshot of `config.yaml` for reproducibility.

## 5) Evaluation & analysis

- Quantitative: evaluate on test set (accuracy, precision/recall, F1, ROC-AUC), produce confusion matrix and classification report.
- Qualitative: use the notebook `notebooks/data_exploration.ipynb` and the explainer module to inspect model attention/heatmaps on samples.

## 6) Explainability & debugging

- Explainer: `src/explainer.py`
  - Attention Rollout for ViT and Grad-CAM style maps are implemented to visualize model focus.
  - Outputs: overlay heatmaps and attention maps to help assess why the model made its decision.

## 7) Inference & demos

- Prediction
  - `src/predict.py` performs single-image or batch inference and returns (label, confidence) and optional explainability outputs.

- Demo apps
  - Streamlit app: `src/app.py` — upload image, get prediction + explanation + basic stats.
  - Gradio demo: `src/demo.py` — quick sharing interface for local testing.

- Launch hints
  - Example commands (after venv and deps installed):
    ```bash
    python src/predict.py --image path/to/image.jpg
    streamlit run src/app.py
    python src/demo.py
    ```

## 8) Utilities & scripts

- `scripts/fetch_data.sh` — checks for dataset at `/home/codespace/datasets/DAIA`, prints sizes, and optionally copies files into the workspace (`--copy-to-workspace`) using `rsync`.
- `scripts/crop_gemini_only.py`, `scripts/test_watermark_crop.py`, `scripts/crop_watermark_batch.py` — watermark handling and testing utilities (they keep backups by default).
- `scripts/setup.py` / `scripts/setup.sh` — helper to create venv and install dependencies (if present).

## 9) Reproducibility & recommended ops

- Repro steps
  1. Create a venv and install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
  2. Ensure datasets exist at `/home/codespace/datasets/DAIA` (or run `scripts/fetch_data.sh`).
  3. Check `config.yaml` for data_dir and hyperparameters.
  4. Run training: `python src/train.py --config config.yaml` (or a smoke test first).

- Dataset versioning options
  - DVC + cloud remote (recommended for large datasets and reproducible pipelines).
  - Git LFS for simpler use-cases.
  - If a large file was committed in the past, remove it from history with `git filter-repo` or BFG (I can help if needed).

## 10) Quick repo map (where things live)

- Config: `config.yaml`
- Data loader & transforms: `src/data_loader.py`
- Model & training loop: `src/model.py`, `src/train.py`
- Prediction & demos: `src/predict.py`, `src/app.py`, `src/demo.py`
- Explainer: `src/explainer.py`
- Scripts: `scripts/` (fetch_data.sh, crop_*.py, setup, test scripts)
- Notebooks: `notebooks/data_exploration.ipynb`
- Docs: `docs/`, `README.md`, `COMMANDS.md`

## 11) Recommended next steps

- Add DVC with a remote if you want dataset versioning and easy sharing.
- Add CI smoke tests: data loader test, one-epoch training smoke test, and a small inference test.
- Add a Dockerfile for reproducible demo deployment.
- Save simple dataset metadata CSV (filename,label,source) alongside images to ease audits and experiments.

---

If you want this placed somewhere else (README root or a different filename), or want the file committed to the current branch with a specific commit message, tell me and I can do that next.
