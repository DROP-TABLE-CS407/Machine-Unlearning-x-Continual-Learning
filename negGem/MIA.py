from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from typing import List, Tuple, Dict, Any, Sequence

# -----------------------------------------------------------------------------
# Helper utilities (loader & forward‑pass collectors)
# -----------------------------------------------------------------------------

def _to_loader(data: np.ndarray | Sequence,
               labels: np.ndarray | Sequence,
               batch: int = 1024,
               device: str | torch.device = "cpu",
               shuffle: bool = False) -> DataLoader:
    """Wrap numpy arrays in a DataLoader living on *device* for fast iter."""
    x = torch.as_tensor(data, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.long,   device=device)
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=shuffle)


def _collect(model, loader, task_idx, op="loss"):
    model.eval()
    outputs = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
            logits = model(x, task_idx)

            if op == "loss":
                loss = F.cross_entropy(logits, y, reduction='none')
                outputs.append(loss.cpu().numpy())
            elif op == "margin":
                top2 = torch.topk(logits, 2, dim=1).values
                margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()
                outputs.append(margin)
            elif op == "confidence":
                conf = torch.softmax(logits, dim=1).max(dim=1).values.cpu().numpy()
                outputs.append(conf)
            else:
                raise ValueError(f"Unsupported op: {op}")
    return np.concatenate(outputs)

# -----------------------------------------------------------------------------
# 1) Basic MIA (loss‑based, SCRUB‑style)
# -----------------------------------------------------------------------------

def basic_mia(model,
              forget_data: np.ndarray, forget_labels: np.ndarray,
              test_data:   np.ndarray, test_labels:   np.ndarray,
              task_idx: int,
              *, device="cpu", clip_val=400.0, cv_splits=5,
              rng: np.random.Generator | None = None) -> Dict[str, float]:
    """Balanced loss‑based MIA using logistic regression."""
    rng = rng or np.random.default_rng()

    # Determine balanced sample size
    n = min(len(forget_data), len(test_data)) // 2
    if n < 2:
        return {"type": "basic", "attack_acc": .5, "cv_acc": .5, "auc": .5, "task": task_idx}

    # Sample 2*n from each to allow train+eval
    idx_f = rng.choice(len(forget_data), 2 * n, replace=False)
    idx_t = rng.choice(len(test_data),   2 * n, replace=False)

    f_data, t_data = forget_data[idx_f], test_data[idx_t]
    f_labels, t_labels = forget_labels[idx_f], test_labels[idx_t]

    # Split into train/eval
    f_train, f_eval = f_data[:n], f_data[n:]
    t_train, t_eval = t_data[:n], t_data[n:]
    f_lab_train, f_lab_eval = f_labels[:n], f_labels[n:]
    t_lab_train, t_lab_eval = t_labels[:n], t_labels[n:]

    # Create loaders
    loaders = {
        "f_train": _to_loader(f_train, f_lab_train, device=device),
        "t_train": _to_loader(t_train, t_lab_train, device=device),
        "f_eval":  _to_loader(f_eval,  f_lab_eval,  device=device),
        "t_eval":  _to_loader(t_eval,  t_lab_eval,  device=device)
    }

    # Collect per-sample loss
    lf_tr = _collect(model, loaders["f_train"], task_idx, op="loss")
    lt_tr = _collect(model, loaders["t_train"], task_idx, op="loss")
    lf_ev = _collect(model, loaders["f_eval"],  task_idx, op="loss")
    lt_ev = _collect(model, loaders["t_eval"],  task_idx, op="loss")

    clip = lambda v: np.clip(v, -clip_val, clip_val)
    lf_tr, lt_tr, lf_ev, lt_ev = map(clip, [lf_tr, lt_tr, lf_ev, lt_ev])

    # Prepare binary classification data
    X_train = np.concatenate([lf_tr, lt_tr]).reshape(-1, 1)
    y_train = np.concatenate([np.ones_like(lf_tr), np.zeros_like(lt_tr)])
    X_eval  = np.concatenate([lf_ev, lt_ev]).reshape(-1, 1)
    y_eval  = np.concatenate([np.ones_like(lf_ev), np.zeros_like(lt_ev)])

    # Manual CV
    split = len(X_train) // cv_splits
    cv_scores = []
    for k in range(cv_splits):
        s = slice(k * split, (k + 1) * split)
        tr_idx = np.r_[0:s.start, s.stop:len(X_train)]
        clf = LogisticRegression(max_iter=1000).fit(X_train[tr_idx], y_train[tr_idx])
        cv_scores.append(accuracy_score(y_train[s], clf.predict(X_train[s])))

    # Final attack classifier
    attacker = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    acc = accuracy_score(y_eval, attacker.predict(X_eval))
    auc = roc_auc_score(y_eval, attacker.predict_proba(X_eval)[:, 1]) if len(np.unique(y_eval)) == 2 else .5

    return {
        "type": "basic",
        "attack_acc": float(acc),
        "cv_acc": float(np.mean(cv_scores)),
        "auc": float(auc),
        "task": task_idx
    }

def run_mia(method: str,
            model,
            forget_data, forget_labels,
            test_data,   test_labels,
            task_idx: int | None,
            device="cpu",
            **kwargs):
    """Unified entry‑point.  *task_idx* can be None to bypass head masking."""
    if method.lower() == "basic":
        return basic_mia(model, forget_data, forget_labels,
                          test_data, test_labels,
                          task_idx=task_idx, device=device,
                          clip_val=kwargs.get("clip_val", 400.),
                          cv_splits=kwargs.get("cv_splits", 5))
    else:
        raise ValueError(f"Unknown MIA method '{method}'")