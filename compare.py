"""
Vergleich: Random Forest vs. CNN für EEG-Bewegungserkennung.

    python compare.py                 # synthetische Daten (schnell)
    python compare.py --physionet     # echte Daten (braucht mne)
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Kein Fenster, nur Dateien speichern.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from eeg.data import CHANNELS, CLASSES, load_physionet, make_synthetic
from eeg.features import band_power, feature_names
from eeg.models import predict_cnn, random_forest, train_cnn


def standardize(train: np.ndarray, test: np.ndarray):
    # Wir normalisieren pro Kanal. Mittelwert und Std kommen NUR aus dem Training.
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True) + 1e-8
    return (train - mean) / std, (test - mean) / std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--physionet", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    # 1) Daten laden
    X, y = load_physionet() if args.physionet else make_synthetic()
    print(f"Daten: {X.shape[0]} Trials, {X.shape[1]} Kanäle, {X.shape[2]} Zeitpunkte")

    out = Path("results")
    out.mkdir(exist_ok=True)
    results = {"rf": {"acc": [], "f1": [], "sec": []}, "cnn": {"acc": [], "f1": [], "sec": []}}
    all_true, rf_pred_all, cnn_pred_all = [], [], []
    importances = np.zeros(len(feature_names(CHANNELS)))

    # 2) Kreuzvalidierung: Wir teilen die Daten in k Teile.
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        print(f"\nFold {fold}/{args.folds}")

        # --- Random Forest ---
        t0 = time.time()
        rf = random_forest()
        rf.fit(band_power(X[tr]), y[tr])
        rf_pred = rf.predict(band_power(X[te]))
        results["rf"]["sec"].append(time.time() - t0)
        importances += rf.feature_importances_

        # --- CNN ---
        # Ein Teil vom Training ist für die Validierung (Early Stopping).
        # Vorher mischen wir die Indizes, sonst ist die Validierung nicht fair.
        shuffled = np.random.default_rng(fold).permutation(tr)
        n_val = len(tr) // 5
        tr_cnn, val_cnn = shuffled[n_val:], shuffled[:n_val]
        Xtr, Xval = standardize(X[tr_cnn], X[val_cnn])
        _, Xte = standardize(X[tr_cnn], X[te])
        t0 = time.time()
        model = train_cnn(Xtr, y[tr_cnn], Xval, y[val_cnn], len(CLASSES), epochs=args.epochs)
        cnn_pred = predict_cnn(model, Xte)
        results["cnn"]["sec"].append(time.time() - t0)

        # Wir speichern die Ergebnisse.
        for name, pred in (("rf", rf_pred), ("cnn", cnn_pred)):
            results[name]["acc"].append(accuracy_score(y[te], pred))
            results[name]["f1"].append(f1_score(y[te], pred, average="macro"))
        all_true.extend(y[te])
        rf_pred_all.extend(rf_pred)
        cnn_pred_all.extend(cnn_pred)
        print(f"  RF  acc={results['rf']['acc'][-1]:.3f}   CNN acc={results['cnn']['acc'][-1]:.3f}")

    # 3) Zusammenfassung
    print("\n=== Ergebnis ===")
    summary = {}
    for name in ("rf", "cnn"):
        acc, f1, sec = (np.array(results[name][k]) for k in ("acc", "f1", "sec"))
        summary[name] = {"accuracy_mean": acc.mean(), "accuracy_std": acc.std(),
                         "f1_macro_mean": f1.mean(), "train_seconds_mean": sec.mean()}
        print(f"{name.upper():4s} Accuracy {acc.mean():.3f} ± {acc.std():.3f} | "
              f"F1 {f1.mean():.3f} | Zeit {sec.mean():.1f}s")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # 4) Bilder: Konfusionsmatrix und wichtigste Features
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pred, title in ((axes[0], rf_pred_all, "Random Forest"), (axes[1], cnn_pred_all, "CNN")):
        ConfusionMatrixDisplay.from_predictions(all_true, pred, display_labels=CLASSES,
                                                ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=150)

    names = feature_names(CHANNELS)
    top = np.argsort(importances)[::-1][:10]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([names[i] for i in top][::-1], importances[top][::-1] / args.folds)
    ax.set_title("Top 10 Features (Random Forest)")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=150)
    print("Bilder gespeichert in results/")


if __name__ == "__main__":
    main()
