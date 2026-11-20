"""
Merkmale (Features) für den Random Forest.
Kommentare auf Deutsch (Niveau A2).
"""
import numpy as np
from scipy.signal import welch

from .data import FS

# Die wichtigen Frequenzbänder für Bewegung.
BANDS = {
    "mu": (8, 13),
    "beta_low": (13, 20),
    "beta_high": (20, 30),
}


def band_power(X: np.ndarray) -> np.ndarray:
    """
    Für jeden Kanal und jedes Band rechnen wir die Leistung aus.
    X: (Trials, Kanäle, Zeit)  ->  Rückgabe: (Trials, Kanäle * Bänder)
    """
    # Welch-Methode: Wir schätzen das Spektrum mit überlappenden Fenstern.
    freqs, psd = welch(X, fs=FS, nperseg=FS, axis=-1)
    feats = []
    for lo, hi in BANDS.values():
        mask = (freqs >= lo) & (freqs < hi)
        # Mittelwert im Band, dann Logarithmus (das macht die Werte stabiler).
        feats.append(np.log(psd[..., mask].mean(axis=-1) + 1e-12))
    return np.concatenate(feats, axis=1)


def feature_names(channels) -> list:
    # Schöne Namen, z.B. "C3_mu". Gut für die Feature-Importance.
    return [f"{ch}_{band}" for band in BANDS for ch in channels]
