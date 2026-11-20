# 🧠 EEG Motor Activity Classifier – CNN vs. Random Forest

Detects **motor imagery (left hand / right hand / rest)** from EEG signals and compares
a classical ML pipeline with a deep-learning model.

| Approach | Input | Idea |
|---|---|---|
| **Random Forest** | Log band power (mu 8–13 Hz, beta 13–30 Hz) per channel, Welch PSD | Hand-crafted, interpretable features |
| **Compact CNN** (EEGNet-style) | Raw, standardized EEG (channels × time) | Learns temporal + spatial filters end-to-end |

## Data
- **Synthetic (default):** physiologically motivated signals – mu/beta rhythms, 1/f noise and
  *event-related desynchronization* (ERD) over C3/C4 contralateral to the imagined hand.
- **Real:** [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) via MNE (`--physionet`).

## Run
```bash
pip install -r requirements.txt
python compare.py               # synthetic data, 5-fold CV
python compare.py --physionet   # real EEG (downloads automatically)
```
Outputs in `results/`: `summary.json`, `confusion_matrices.png`, `feature_importance.png`.

## Methodology highlights
- Stratified k-fold cross-validation, standardization fitted **only on training folds** (no leakage)
- Early stopping on an inner validation split
- Feature importance shows the RF relies on **C3/C4 mu power** – matching neuroscience expectations

## Skills
Python · signal processing (Welch PSD, band power) · scikit-learn · PyTorch · CNNs · MNE · evaluation design
