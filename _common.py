"""Shared helpers: dataset loading, backbones, focal loss, metrics."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA, RESULTS, CKPTS = ROOT / "data", ROOT / "results", ROOT / "checkpoints"
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
SEED = 1337


def set_seed(seed: int = SEED) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_index(qatar_root: Path) -> pd.DataFrame:
    """Build a (filepath, label, source) frame for the Qatar TB-CXR corpus.

    The Kaggle archive unpacks into a folder containing 'Normal/' and
    'Tuberculosis/' (capitalisation varies across mirrors). Both are walked.
    """
    rows = []
    for cls in qatar_root.rglob("*"):
        if not cls.is_dir():
            continue
        name = cls.name.lower()
        if name in {"normal"}:
            label = 0
        elif name.startswith("tuberculosis") or name in {"tb"}:
            label = 1
        else:
            continue
        for p in cls.glob("*"):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                rows.append({"path": str(p), "label": label, "source": "qatar"})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f"No images found under {qatar_root}. "
                           "Download the Qatar TB-CXR v1 corpus from "
                           "https://www.kaggle.com/datasets/tawsifurrahman/"
                           "tuberculosis-tb-chest-xray-dataset and unpack "
                           "into this folder.")
    return df


def make_splits(df: pd.DataFrame, train=0.8, val=0.1, test=0.1,
                seed=SEED, subsample_per_class: int | None = None):
    """Stratified train/val/test split, optionally subsampled per class.

    subsample_per_class: if set, keep at most N images per class BEFORE
    splitting. Use to run a fast configuration (e.g. 800 per class -> 1600
    total images, ~5 min training on GPU) that still gives meaningful
    metrics on a held-out test split of ~160 images.
    """
    assert abs(train + val + test - 1.0) < 1e-6
    if subsample_per_class is not None:
        parts = []
        for label, sub in df.groupby("label"):
            parts.append(sub.sample(n=min(subsample_per_class, len(sub)),
                                    random_state=seed))
        df = pd.concat(parts, ignore_index=True)
    out = {"train": [], "val": [], "test": []}
    for label, sub in df.groupby("label"):
        idx = sub.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(idx); n_tr = int(n * train); n_va = int(n * val)
        out["train"].append(idx.iloc[:n_tr])
        out["val"].append(idx.iloc[n_tr:n_tr + n_va])
        out["test"].append(idx.iloc[n_tr + n_va:])
    return {k: pd.concat(v, ignore_index=True).sample(frac=1, random_state=seed)
            for k, v in out.items()}


class CXRDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, train: bool):
        self.frame = frame.reset_index(drop=True)
        if train:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self): return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        return self.tf(img), int(row["label"])


def loader(df: pd.DataFrame, train: bool, batch_size: int = 32) -> DataLoader:
    return DataLoader(CXRDataset(df, train), batch_size=batch_size,
                      shuffle=train, num_workers=0, pin_memory=True)


# -- Models ---------------------------------------------------------------
def build_backbone(name: str, num_classes: int = 2) -> nn.Module:
    """Construct an ImageNet-pretrained backbone with a 2-way head."""
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone {name}")
    return m


def n_params(m: nn.Module) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6


# -- Losses ---------------------------------------------------------------
class FocalLoss(nn.Module):
    """Binary focal loss as in Lin et al. 2020, formulated for 2-class logits."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        # per-sample probability of the true class
        pt = p.gather(1, target.view(-1, 1)).squeeze(1).clamp_min(1e-8)
        logpt = logp.gather(1, target.view(-1, 1)).squeeze(1)
        alpha_t = torch.where(target == 1, torch.full_like(pt, self.alpha),
                              torch.full_like(pt, 1 - self.alpha))
        loss = -alpha_t * (1 - pt) ** self.gamma * logpt
        return loss.mean()


# -- Metrics --------------------------------------------------------------
def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Accuracy, AUROC, Sens@90% specificity. probs is P(TB)=probs[:,1]."""
    p1 = probs[:, 1]
    pred = (p1 >= 0.5).astype(int)
    acc = (pred == labels).mean()
    auc = roc_auc_score(labels, p1)
    sens90 = sens_at_spec(labels, p1, target_spec=0.90)
    cm = confusion_matrix(labels, pred).tolist()
    return {"accuracy": float(acc), "auroc": float(auc),
            "sens_at_90_spec": float(sens90), "confusion_matrix": cm}


def sens_at_spec(y: np.ndarray, score: np.ndarray, target_spec: float = 0.90) -> float:
    """Sensitivity at first threshold where specificity >= target_spec."""
    fpr, tpr, thr = roc_curve(y, score)
    spec = 1 - fpr
    ok = np.where(spec >= target_spec)[0]
    if len(ok) == 0:
        return float("nan")
    # take the threshold with highest sensitivity among those satisfying spec
    return float(tpr[ok].max())


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error, equal-mass binning, on the *predicted-class*
    confidence (per Naeini 2015 / Guo 2017)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    # equal-mass bins on confidence
    order = np.argsort(conf)
    conf, correct = conf[order], correct[order]
    edges = np.linspace(0, len(conf), n_bins + 1, dtype=int)
    val = 0.0
    for i in range(n_bins):
        a, b = edges[i], edges[i + 1]
        if b <= a: continue
        bin_conf = conf[a:b].mean()
        bin_acc  = correct[a:b].mean()
        val += (b - a) / len(conf) * abs(bin_acc - bin_conf)
    return float(val)


def reliability_bins(probs: np.ndarray, labels: np.ndarray,
                     n_bins: int = 10, equal_mass: bool = True):
    """Return (mean_conf_per_bin, accuracy_per_bin, weight_per_bin).

    equal_mass=True: each bin holds the same number of samples (good when
    confidences cluster — typical for well-trained classifiers).
    equal_mass=False: classic equal-width bins on [0,1].
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    if equal_mass:
        order = np.argsort(conf)
        conf_s, corr_s = conf[order], correct[order]
        edges = np.linspace(0, len(conf_s), n_bins + 1, dtype=int)
        out = []
        for i in range(n_bins):
            a, b = edges[i], edges[i + 1]
            if b <= a: continue
            out.append((float(conf_s[a:b].mean()),
                        float(corr_s[a:b].mean()),
                        int(b - a)))
        return out
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < n_bins - 1
                                  else conf <= edges[i + 1])
        if m.sum() == 0: continue
        out.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return out


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path: Path):
    return json.loads(Path(path).read_text())
