"""Shared, auditable utilities for the modern Liu2024 experiment notebooks."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LIU29 = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3", "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"]
MOTOR13 = ["F3", "F4", "FCz", "FC3", "FC4", "Cz", "C3", "C4", "CP3", "CP4", "Pz", "P3", "P4"]
PRIMARY_MODELS = ["FBCNet", "EEGTCNet", "IFNet", "FBMSNet", "EEGNet", "ShallowFBCSPNet"]
OPTIONAL_MODELS = ["ATCNet", "EEGConformer", "EEGNeX"]
LIGHTWEIGHT_MODELS = ["FBLightConvNet", "SincShallowNet", "EEGITNet"]
LIGHTWEIGHT_OPTIONAL_MODELS = ["EEGInceptionMI", "MSVTNet", "CTNet"]


def stable_hash(payload, length=16):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:length]


def array_hash(*arrays, length=16):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()[:length]


def subject_id(path):
    return path.parent.name


def locate_subject_files(root, subjects=None):
    paths = sorted(Path(root).glob("sub-*/sub-*_task-motor-imagery_eeg.mat"))
    if subjects is not None:
        wanted = {f"sub-{int(s):02d}" if str(s).isdigit() else str(s) for s in subjects}
        paths = [p for p in paths if subject_id(p) in wanted]
    if not paths:
        raise FileNotFoundError(f"No Liu2024 MAT files found under {root}")
    return paths


def load_subject(path, cfg):
    eeg = loadmat(path)["eeg"][0, 0]
    raw = np.asarray(eeg["rawdata"], dtype=np.float32)
    y = np.asarray(eeg["label"]).reshape(-1).astype(int) - 1
    marker = raw[:, int(cfg["marker_channel_index"]), :]
    plausible = cfg["onset_plausible_range"]
    candidate_onsets = []
    for row in marker:
        hits = np.flatnonzero(row == cfg["onset_marker_value"])
        hits = hits[(hits >= plausible[0]) & (hits <= plausible[1])]
        candidate_onsets.append(int(hits[0]) if len(hits) else None)
    valid_onsets = [onset for onset in candidate_onsets if onset is not None]
    if not valid_onsets:
        raise ValueError(f"No plausible MI markers found in {path}")
    fallback = int(round(float(np.median(valid_onsets))))
    onsets = [onset if onset is not None else fallback for onset in candidate_onsets]
    names = LIU29
    keep = list(range(17)) + list(range(18, 30))
    x = raw[:, keep, :]
    selected = cfg["channel_set"]
    if selected == "motor13":
        idx = [names.index(ch) for ch in MOTOR13]
        x, names = x[:, idx], MOTOR13
    elif selected != "liu29":
        raise ValueError(f"Unknown channel_set={selected!r}")
    sos = signal.butter(int(cfg["filter_order"]), cfg["bandpass_hz"], btype="bandpass", fs=cfg["native_sfreq"], output="sos")
    # Each complete 8 s trial is filtered independently, then marker-relative cropped.
    x = signal.sosfiltfilt(sos, x, axis=-1)
    n_native = int(round(cfg["window_seconds"] * cfg["native_sfreq"]))
    x = np.stack([trial[:, onset:onset + n_native] for trial, onset in zip(x, onsets)])
    if x.shape[-1] != n_native:
        raise ValueError(f"Incomplete marker-relative crop in {path}")
    n_target = int(round(cfg["window_seconds"] * cfg["target_sfreq"]))
    x = signal.resample(x, n_target, axis=-1).astype(np.float32)
    if len(x) != 40 or set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Expected 40 balanced binary trials in {path}; got {len(x)}, labels={np.unique(y)}")
    return x, y, names, np.asarray(onsets)


def make_splits(y, cfg, sid):
    splitter = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=cfg["cv_random_state"])
    splits = [{"fold_id": i, "train_indices": tr.tolist(), "test_indices": te.tolist()} for i, (tr, te) in enumerate(splitter.split(np.zeros(len(y)), y))]
    seen = sorted(i for fold in splits for i in fold["test_indices"])
    if seen != list(range(len(y))):
        raise AssertionError(f"Every {sid} trial must occur exactly once in outer test folds")
    return splits


def fit_normalizer(x_train, mode, eps=1e-6):
    if mode == "channel_standardize":
        mean = x_train.mean(axis=(0, 2), keepdims=True); scale = x_train.std(axis=(0, 2), keepdims=True)
    elif mode == "scalar_standardize":
        mean = np.asarray(x_train.mean(), dtype=np.float32); scale = np.asarray(x_train.std(), dtype=np.float32)
    elif mode == "exponential_standardize":
        # Fold-local channel statistics; no temporal state crosses trial boundaries.
        mean = x_train.mean(axis=(0, 2), keepdims=True); scale = x_train.std(axis=(0, 2), keepdims=True)
    else:
        raise ValueError(f"Unsupported normalization_mode={mode!r}")
    return mean, np.maximum(scale, eps)


def build_model(name, n_chans, n_times, sfreq, device="cpu", model_kwargs=None):
    from braindecode import models
    allowed = PRIMARY_MODELS + OPTIONAL_MODELS + LIGHTWEIGHT_MODELS + LIGHTWEIGHT_OPTIONAL_MODELS
    if name not in allowed:
        raise ValueError(f"Unknown model {name}; allowed={allowed}")
    cls = getattr(models, name)
    kwargs = {"n_chans": n_chans, "n_outputs": 2, "n_times": n_times, "sfreq": sfreq}
    kwargs.update(model_kwargs or {})
    if name == "FBLightConvNet":
        win_len = int(kwargs.get("win_len", 256))
        if n_times % win_len:
            raise ValueError(f"FBLightConvNet win_len={win_len} must divide n_times={n_times} exactly")
        kwargs["win_len"] = win_len
    try:
        model = cls(**kwargs).to(device)
        with torch.no_grad():
            out = model(torch.zeros(2, n_chans, n_times, device=device))
        dense_shallow = name == "ShallowFBCSPNet" and kwargs.get("final_conv_length") == 1
        valid_dense = dense_shallow and out.ndim >= 3 and tuple(out.shape[:2]) == (2, 2)
        if tuple(out.shape) != (2, 2) and not valid_dense:
            raise RuntimeError(f"expected (2,2) logits, got {tuple(out.shape)}")
    except Exception as exc:
        raise RuntimeError(f"{name} incompatible with input ({n_chans}, {n_times}) at {sfreq} Hz and kwargs={kwargs}: {exc}") from exc
    return model


def parameter_count(model, trainable_only=False):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only))


def identify_classifier(model):
    candidates = [(name, module) for name, module in model.named_modules() if name and isinstance(module, nn.Linear)]
    candidates += [(name, module) for name, module in model.named_modules() if name and isinstance(module, (nn.Conv1d, nn.Conv2d)) and getattr(module, "out_channels", None) == 2]
    if not candidates:
        raise RuntimeError(f"Could not identify classifier head for {type(model).__name__}")
    return candidates[-1]


def configure_adaptation(model, mode):
    for p in model.parameters():
        p.requires_grad = True
    head_name, head = identify_classifier(model)
    if mode == "frozen_encoder":
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    elif mode not in {"target_only", "full_finetune", "source_pretrain"}:
        raise ValueError(f"Unknown adaptation mode {mode}")
    return head_name, parameter_count(model, True)


def augment_training_batch(x, cfg):
    augmentation = cfg.get("augmentation") or {"enabled": False}
    if not augmentation.get("enabled", False):
        return x
    transformed = x
    for spec in augmentation.get("transforms", []):
        probability = float(spec.get("probability", 1.0))
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Augmentation probability must be in [0,1], got {probability}")
        apply_mask = (torch.rand(len(transformed), 1, 1, device=transformed.device) < probability).to(transformed.dtype)
        name = spec["name"]
        if name == "relative_gaussian_noise":
            std_fraction = float(spec["std_fraction"])
            if std_fraction < 0:
                raise ValueError("std_fraction must be non-negative")
            channel_scale = transformed.std(dim=-1, keepdim=True, unbiased=False).clamp_min(float(cfg.get("normalization_eps", 1e-6)))
            transformed = transformed + apply_mask * std_fraction * channel_scale * torch.randn_like(transformed)
        elif name == "amplitude_scale":
            low, high = map(float, spec["interval"])
            if not 0 < low <= high:
                raise ValueError(f"Invalid amplitude interval: {[low, high]}")
            factors = low + (high - low) * torch.rand(len(transformed), 1, 1, device=transformed.device)
            transformed = transformed * (1.0 + apply_mask * (factors - 1.0))
        else:
            raise ValueError(f"Unsupported training augmentation {name!r}")
    return transformed


def train_eval(model, x_train, y_train, x_test, y_test, cfg, seed, lr=None, epochs=None, batch_size=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = next(model.parameters()).device
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).long()), batch_size=batch_size or cfg["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr or cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    start = time.perf_counter(); model.train()
    for _ in range(int(epochs or cfg["n_epochs"])):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            xb = augment_training_batch(xb.to(device), cfg)
            loss = nn.functional.cross_entropy(model(xb), yb.to(device))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"]); optimizer.step()
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        prob = model(torch.from_numpy(x_test).to(device)).softmax(1).cpu().numpy()
    pred = prob.argmax(1)
    return pred, prob, elapsed


def fold_result(sid, fold_id, test_idx, y, pred, prob, model, elapsed, seed, extra=None):
    hist = np.bincount(pred, minlength=2)
    result = {"subject_id": sid, "fold_id": int(fold_id), "test_indices": list(map(int, test_idx)), "true_labels": y.tolist(), "predictions": pred.tolist(), "probabilities": prob.tolist(), "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(), "prediction_histogram": hist.tolist(), "collapse_diagnostics": {"collapsed": bool(hist.max() / len(pred) >= 0.9)}, "parameter_count": parameter_count(model), "trainable_parameter_count": parameter_count(model, True), "training_seconds": float(elapsed), "seed": int(seed)}
    result.update(extra or {})
    return result


def checkpoint_signature(cfg, source_ids, model_name):
    keys = [
        "channel_set", "native_sfreq", "target_sfreq", "marker_channel_index",
        "onset_marker_value", "onset_plausible_range", "window_seconds",
        "bandpass_hz", "filter_order", "normalization_mode", "normalization_eps",
        "model_kwargs", "source_epochs", "source_batch_size", "batch_size", "learning_rate",
        "weight_decay", "gradient_clip_norm", "seed",
    ]
    versions = {
        package: importlib.metadata.version(package)
        for package in ("braindecode", "torch")
    }
    payload = {
        "model": model_name,
        "sources": sorted(source_ids),
        "config": {key: cfg.get(key) for key in keys},
        "software_versions": versions,
    }
    return stable_hash(payload)


def assert_source_exclusion(target_id, source_ids):
    if target_id in source_ids:
        raise AssertionError(f"LOSO violation: target {target_id} appears in source IDs")


def save_checkpoint(path, model, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_compatible_checkpoint(path, expected):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("metadata") != expected:
        raise RuntimeError(f"Refusing incompatible checkpoint {path}; metadata mismatch")
    return payload["state_dict"]


def bootstrap_ci(values, seed=2026, n_boot=10000):
    values = np.asarray(values, float); rng = np.random.default_rng(seed)
    means = rng.choice(values, (n_boot, len(values)), replace=True).mean(1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]
