"""Unit-Tests für Daten, Merkmale und Modell (schnell, ohne Internet)."""
import numpy as np
import torch

from eeg.data import CHANNELS, CLASSES, FS, make_synthetic
from eeg.features import BANDS, band_power, feature_names
from eeg.models import CompactEEGNet, random_forest


def test_synthetische_daten_haben_die_richtige_form():
    X, y = make_synthetic(n_trials=12, seconds=1.0, seed=1)
    assert X.shape == (12, len(CHANNELS), FS)
    assert X.dtype == np.float32
    assert set(np.unique(y)) <= set(range(len(CLASSES)))


def test_gleicher_seed_gleiche_daten():
    a, ya = make_synthetic(n_trials=5, seconds=0.5, seed=7)
    b, yb = make_synthetic(n_trials=5, seconds=0.5, seed=7)
    assert np.array_equal(a, b) and np.array_equal(ya, yb)


def test_band_power_form_und_namen():
    X, _ = make_synthetic(n_trials=4, seconds=1.0, seed=2)
    feats = band_power(X)
    assert feats.shape == (4, len(CHANNELS) * len(BANDS))
    assert np.isfinite(feats).all()
    assert len(feature_names(CHANNELS)) == feats.shape[1]


def test_erd_ist_in_den_merkmalen_sichtbar():
    # Linke Hand -> Mu-Rhythmus an C4 schwächer als in Ruhe (Physiologie aus data.py)
    X, y = make_synthetic(n_trials=200, seconds=2.0, noise=0.5, seed=3)
    feats = band_power(X)
    names = feature_names(CHANNELS)
    c4_mu = feats[:, names.index("C4_mu")]
    assert c4_mu[y == 1].mean() < c4_mu[y == 0].mean()


def test_random_forest_lernt_die_synthetischen_daten():
    X, y = make_synthetic(n_trials=240, seconds=1.0, noise=0.5, seed=4)
    feats = band_power(X)
    model = random_forest(seed=0).fit(feats[:180], y[:180])
    assert model.score(feats[180:], y[180:]) > 0.7  # Zufall wäre ca. 0,33


def test_cnn_ausgabe_form():
    model = CompactEEGNet(n_channels=len(CHANNELS), n_samples=FS, n_classes=len(CLASSES))
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(3, len(CHANNELS), FS))
    assert out.shape == (3, len(CLASSES))
