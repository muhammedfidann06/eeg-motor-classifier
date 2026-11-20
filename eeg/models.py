"""
Zwei Modelle: Random Forest (klassisch) und ein kleines CNN (Deep Learning).
Kommentare auf Deutsch (Niveau A2).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn


def random_forest(seed: int = 0) -> RandomForestClassifier:
    # 300 Bäume. Jeder Baum sieht nur einen Teil der Daten.
    return RandomForestClassifier(n_estimators=300, max_depth=None,
                                  n_jobs=-1, random_state=seed)


class CompactEEGNet(nn.Module):
    """
    Ein kleines CNN, inspiriert von EEGNet (Lawhern et al., 2018).

    1) Zeitliche Faltung: Das Netz lernt Frequenzfilter.
    2) Räumliche Faltung (depthwise): Das Netz lernt, welche Kanäle wichtig sind.
    3) Separable Faltung + Pooling: weniger Parameter, weniger Overfitting.
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int,
                 f1: int = 8, depth: int = 2, f2: int = 16, dropout: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            # Filter über die Zeit (Kernel 64 = ca. 0,4 s bei 160 Hz).
            nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
            # Filter über die Kanäle (für jeden Zeitfilter extra).
            nn.Conv2d(f1, f1 * depth, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * depth, f1 * depth, (1, 16), padding=(0, 8),
                      groups=f1 * depth, bias=False),
            nn.Conv2d(f1 * depth, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        # Wir testen einmal mit Nullen, wie groß die Ausgabe ist.
        with torch.no_grad():
            n_flat = self._features(torch.zeros(1, 1, n_channels, n_samples)).shape[1]
        self.classifier = nn.Linear(n_flat, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x hat die Form (Batch, Kanäle, Zeit). Wir brauchen noch eine Dimension.
        return self.classifier(self._features(x.unsqueeze(1)))


def train_cnn(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              n_classes: int, epochs: int = 60, lr: float = 1e-3, batch: int = 32,
              seed: int = 0, verbose: bool = True) -> CompactEEGNet:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CompactEEGNet(X_train.shape[1], X_train.shape[2], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.long).to(device)

    best_acc, best_state, patience = 0.0, None, 0
    for ep in range(epochs):
        model.train()
        # Jede Epoche mischen wir die Daten neu.
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        # Wir prüfen das Modell mit den Validierungsdaten.
        model.eval()
        with torch.no_grad():
            acc = (model(Xv).argmax(1) == yv).float().mean().item()
        if verbose and ep % 10 == 0:
            print(f"  Epoche {ep:3d}  loss={loss.item():.3f}  val_acc={acc:.3f}")

        # Early Stopping: Wenn es 15 Epochen nicht besser wird, hören wir auf.
        if acc > best_acc:
            best_acc, patience = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 15:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu()


def predict_cnn(model: CompactEEGNet, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()
