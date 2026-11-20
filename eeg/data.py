"""
EEG-Daten: synthetisch (schnell) oder echt (PhysioNet, mit MNE).
"""
from __future__ import annotations

import numpy as np

FS = 160            # Abtastrate in Hz (wie bei PhysioNet)
CHANNELS = ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4"]
CLASSES = ["rest", "left_hand", "right_hand"]


def make_synthetic(
    n_trials: int = 600,
    seconds: float = 2.0,
    noise: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wir bauen künstliche EEG-Signale.

    Idee (echte Physiologie): Wenn man die linke Hand bewegt, wird der
    Mu-Rhythmus (8–12 Hz) rechts im Gehirn (C4) schwächer. Bei der rechten
    Hand ist es links (C3). Das heißt "Desynchronisation" (ERD).

    Rückgabe: X mit Form (Trials, Kanäle, Zeit) und y mit Labels.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(FS * seconds)
    t = np.arange(n_samples) / FS
    X = np.zeros((n_trials, len(CHANNELS), n_samples), dtype=np.float32)
    y = rng.integers(0, len(CLASSES), size=n_trials)

    c3, c4 = CHANNELS.index("C3"), CHANNELS.index("C4")

    for i in range(n_trials):
        for ch in range(len(CHANNELS)):
            # Jeder Kanal hat einen Mu-Rhythmus und einen Beta-Rhythmus.
            mu_freq = rng.uniform(9, 12)
            beta_freq = rng.uniform(18, 25)
            mu_amp = 2.0
            beta_amp = 0.8

            # Hier machen wir die ERD: weniger Amplitude auf der anderen Seite.
            if y[i] == 1 and ch == c4:      # linke Hand -> C4 schwächer
                mu_amp *= 0.4
                beta_amp *= 0.6
            if y[i] == 2 and ch == c3:      # rechte Hand -> C3 schwächer
                mu_amp *= 0.4
                beta_amp *= 0.6

            phase = rng.uniform(0, 2 * np.pi, size=2)
            signal = (
                mu_amp * np.sin(2 * np.pi * mu_freq * t + phase[0])
                + beta_amp * np.sin(2 * np.pi * beta_freq * t + phase[1])
            )
            # Rosa Rauschen (1/f) ist realistischer als weißes Rauschen.
            X[i, ch] = signal + noise * _pink_noise(n_samples, rng)
    return X, y


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    # Wir filtern weißes Rauschen im Frequenzraum mit 1/sqrt(f).
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    pink = np.fft.irfft(spec / np.sqrt(freqs), n)
    return pink / pink.std()


def load_physionet(subjects=(1, 2, 3), runs=(4, 8, 12)) -> tuple[np.ndarray, np.ndarray]:
    """
    Echte Daten: PhysioNet EEG Motor Movement/Imagery Dataset.
    Runs 4, 8, 12 = Vorstellung: linke oder rechte Hand.
    Braucht: pip install mne  (die Daten werden automatisch geladen)
    """
    import mne
    from mne.datasets import eegbci

    raws = []
    for s in subjects:
        for f in eegbci.load_data(s, list(runs)):
            raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
            eegbci.standardize(raw)
            raws.append(raw)
    raw = mne.concatenate_raws(raws)
    raw.pick(CHANNELS)
    # Wir filtern zwischen 7 und 30 Hz. Dort sind Mu und Beta.
    raw.filter(7.0, 30.0, verbose="ERROR")

    events, _ = mne.events_from_annotations(raw, event_id=dict(T0=0, T1=1, T2=2), verbose="ERROR")
    epochs = mne.Epochs(raw, events, tmin=0.5, tmax=2.5 - 1 / FS, baseline=None,
                        preload=True, verbose="ERROR")
    return epochs.get_data().astype(np.float32), epochs.events[:, 2]
