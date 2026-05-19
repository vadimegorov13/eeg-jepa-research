"""
Liu2024 Source .mat SignalJEPA PreLocal — Improved Within-Subject CV (v2)

Key improvements over the original notebook:
- strategy="full" (warm up new layers for warmup_epochs, then fine-tune all)
- AdamW with weight_decay for better regularisation
- CosineAnnealingLR schedule for smoother convergence
- Early stopping with patience=15
- More epochs (100 vs 50)
- Smaller batch_size (16) → more gradient updates per epoch
- Gaussian noise augmentation to help with the small per-subject dataset (40 trials)
- Lower learning rate (5e-4 vs 1e-3) for stable full fine-tuning

Baseline (original notebook): mean acc = 0.4925 ± 0.1643  (chance level; model did not train)
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import re
import sys
import json
import math
import hashlib
import random
import builtins
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, Subset

from scipy.io import loadmat

import mne
mne.set_log_level("WARNING")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit
from torch.optim.lr_scheduler import CosineAnnealingLR

from braindecode import EEGClassifier
from braindecode.models import SignalJEPA_PreLocal

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

# Script working directory: use parent of this file if it's placed in src/
WORKING_DIR = Path(__file__).resolve().parent

CONFIG = {
    # Paths
    "artifact_dir": str(WORKING_DIR / "artifacts" / "liu2024-source-mat-sjepa-prelocal-v2"),
    "manual_source_extract_dir": None,

    # Dataset
    "dataset_name": "Liu2024_SourceMAT_v2",
    "labels_to_keep": ["left_hand", "right_hand"],
    "subjects_to_use": None,
    "exclude_subjects": [],

    # Source data conventions (unchanged)
    "source_sfreq": 500,
    "source_trial_duration_s": 8.0,
    "drop_source_reference_channel": True,
    "drop_eog_and_marker": True,
    "source_data_scale": 1.0,

    # S-JEPA preprocessing (unchanged)
    "sfreq": 128,
    "bandpass_low": 0.5,
    "bandpass_high": 40.0,
    "average_reference_before_resample_filter": True,

    # S-JEPA window (unchanged)
    "target_window_samples": 537,
    "mi_window_start_s": 2.0,

    # Model (unchanged)
    "model_name": "SignalJEPA_PreLocal",
    "pretrained_mode": "from_pretrained",
    "pretrained_repo_id": "braindecode/signal-jepa_without-chans",

    # --- KEY IMPROVEMENT 1: full fine-tuning strategy ---
    "strategy": "full",   # was "new" → frozen backbone caused zero gradient
    "warmup_epochs": 10,  # warmup: only spatial_conv + final_layer

    # Cross-validation (unchanged)
    "cv_folds": 5,
    "val_split": 0.2,

    # --- KEY IMPROVEMENT 2: training hyperparameters ---
    "batch_size": 16,       # was 32; smaller = more updates/epoch on tiny dataset
    "n_epochs": 100,        # was 50; more time to converge
    "early_stopping_patience": 15,   # was None
    "learning_rate": 5e-4,  # was 1e-3; lower for stable full fine-tuning

    # --- KEY IMPROVEMENT 3: AdamW with weight decay ---
    "optimizer_weight_decay": 1e-4,

    # --- KEY IMPROVEMENT 4: cosine LR schedule ---
    "use_lr_scheduler": True,
    "lr_min": 1e-6,

    # --- KEY IMPROVEMENT 5: Gaussian noise augmentation ---
    "augmentation_noise_fraction": 0.05,  # noise = fraction * per-channel std

    # Reproducibility
    "seed": 12,
    "set_seed": True,

    # Diagnostics
    "extract_spatial_conv_weights": True,
    "collapse_threshold": 0.90,
    "log_spatial_update_stats": True,
    "log_probability_diagnostics": True,
}

SOURCE_EEG_CHANNEL_INDICES_30 = list(range(30))
SOURCE_REFERENCE_INDEX = 17
SOURCE_EEG_CHANNEL_INDICES_29 = [i for i in SOURCE_EEG_CHANNEL_INDICES_30 if i != SOURCE_REFERENCE_INDEX]

SOURCE_EEG_CHANNEL_NAMES_30 = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4",
    "FT7", "FT8", "Cz", "C3", "C4", "T7", "T8", "CPz",
    "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "P7", "P8", "Oz", "O1", "O2",
]

if CONFIG["drop_source_reference_channel"]:
    EEG_CHANNEL_INDICES = SOURCE_EEG_CHANNEL_INDICES_29
    EEG_CHANNEL_NAMES = [n for i, n in enumerate(SOURCE_EEG_CHANNEL_NAMES_30) if i != SOURCE_REFERENCE_INDEX]
else:
    EEG_CHANNEL_INDICES = SOURCE_EEG_CHANNEL_INDICES_30
    EEG_CHANNEL_NAMES = SOURCE_EEG_CHANNEL_NAMES_30

TARGET_N_CLASSES = len(CONFIG["labels_to_keep"])
WINDOW_SAMPLES = int(CONFIG["target_window_samples"])
TARGET_TRIAL_DURATION_S = WINDOW_SAMPLES / float(CONFIG["sfreq"])
MI_WINDOW_START_SAMPLE = int(round(float(CONFIG["mi_window_start_s"]) * float(CONFIG["sfreq"])))
MI_WINDOW_STOP_SAMPLE = MI_WINDOW_START_SAMPLE + WINDOW_SAMPLES

# ---------------------------------------------------------------------------
# 2. Logging
# ---------------------------------------------------------------------------

def create_run_id():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    config_str = json.dumps(CONFIG, sort_keys=True, default=str)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return f"{timestamp}_{config_hash}"

RUN_ID = create_run_id()
ARTIFACT_DIR = Path(CONFIG["artifact_dir"]) / CONFIG["dataset_name"] / RUN_ID
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = ARTIFACT_DIR / "run.log"
_LOG_FILE_HANDLE = open(LOG_PATH, "a", buffering=1, encoding="utf-8", errors="replace")

def _safe_write_text(stream, text):
    try:
        stream.write(text)
        return
    except UnicodeEncodeError:
        pass
    encoding = getattr(stream, "encoding", None) or "utf-8"
    safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    stream.write(safe_text)

def _timestamped_print(*args, **kwargs):
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    file = kwargs.pop("file", None)
    flush = kwargs.pop("flush", False)
    msg = sep.join(str(a) for a in args)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}{end}"
    _safe_write_text(_LOG_FILE_HANDLE, line)
    _LOG_FILE_HANDLE.flush()
    if file is None:
        _safe_write_text(sys.__stdout__, line)
        if flush:
            sys.__stdout__.flush()
    else:
        _safe_write_text(file, line)
        if flush:
            file.flush()

builtins.print = _timestamped_print

print(f"Run ID:     {RUN_ID}")
print(f"Artifacts:  {ARTIFACT_DIR}")
with open(ARTIFACT_DIR / "config.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

# ---------------------------------------------------------------------------
# 3. Reproducibility
# ---------------------------------------------------------------------------

def resolve_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = resolve_device()
print(f"Using device: {DEVICE}")

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

BASE_SEED = int(CONFIG["seed"]) if CONFIG["seed"] is not None else None
if CONFIG["set_seed"]:
    seed_everything(BASE_SEED)
    print(f"Seed initialized: {BASE_SEED}")

# ---------------------------------------------------------------------------
# 4. Data loading helpers (identical to original)
# ---------------------------------------------------------------------------

def find_source_mat_files(root):
    root = Path(root)
    return sorted(root.rglob("*.mat")) if root.exists() else []

def candidate_source_dirs():
    candidates = []
    if CONFIG["manual_source_extract_dir"] is not None:
        candidates.append(Path(CONFIG["manual_source_extract_dir"]))

    search_roots = [WORKING_DIR, WORKING_DIR.parent, WORKING_DIR.parent.parent]

    # Also look in sibling repos (e.g. main EEG_JEPA repo alongside worktree)
    for ancestor in [WORKING_DIR.parent, WORKING_DIR.parent.parent, WORKING_DIR.parent.parent.parent]:
        if ancestor.exists():
            for sibling in ancestor.iterdir():
                if sibling.is_dir() and "EEG_JEPA" in sibling.name and "worktree" not in sibling.name.lower():
                    search_roots.extend([sibling, sibling / "src"])

    for base in search_roots:
        candidates.extend([
            base / "liu2024_figshare" / "sourcedata",
            base / "liu2024_figshare" / "sourcedata" / "sourcedata",
            base / "src" / "liu2024_figshare" / "sourcedata",
            base / "src" / "liu2024_figshare" / "sourcedata" / "sourcedata",
        ])
    seen, uniq = set(), []
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq

def subject_id_from_path(path):
    s = str(path)
    m = re.search(r"sub[-_ ]?(\d{1,2})", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", Path(path).stem)
    if nums:
        return int(nums[-1])
    raise ValueError(f"Could not infer subject id from path: {path}")

def _is_mat_struct(x):
    return hasattr(x, "_fieldnames")

def _walk_mat_object(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).startswith("__"):
                continue
            name = f"{prefix}.{k}" if prefix else str(k)
            yield name, v
            yield from _walk_mat_object(v, name)
    elif _is_mat_struct(obj):
        for k in obj._fieldnames:
            v = getattr(obj, k)
            name = f"{prefix}.{k}" if prefix else str(k)
            yield name, v
            yield from _walk_mat_object(v, name)
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object and obj.size == 1:
            yield from _walk_mat_object(obj.item(), prefix)
        elif obj.dtype == object:
            for idx, item in np.ndenumerate(obj):
                yield from _walk_mat_object(item, f"{prefix}{idx}")

def _normalize_rawdata_shape(rawdata):
    arr = np.asarray(rawdata)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D rawdata, got shape={arr.shape}")
    if arr.shape[0] in (39, 40) and arr.shape[1] >= 30 and arr.shape[2] >= 1000:
        return arr
    trial_axes = [i for i, s in enumerate(arr.shape) if s in (39, 40)]
    if trial_axes and trial_axes[0] != 0:
        arr = np.moveaxis(arr, trial_axes[0], 0)
    time_axis = int(np.argmax(arr.shape))
    if time_axis != 2:
        arr = np.moveaxis(arr, time_axis, 2)
    if arr.shape[1] < 30:
        raise ValueError(f"Could not normalize rawdata to trials x channels x samples, got {arr.shape}")
    return arr

def load_subject_mat(path):
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    arrays = [(name, np.asarray(value)) for name, value in _walk_mat_object(mat)
              if isinstance(value, np.ndarray) and value.dtype != object]
    raw_candidates, label_candidates = [], []
    for name, arr in arrays:
        lname = name.lower()
        if arr.ndim == 3:
            score = 0
            if "rawdata" in lname or "data" in lname:
                score += 10
            if 39 <= min(arr.shape) <= 40 or arr.shape[0] in (39, 40):
                score += 3
            if max(arr.shape) >= 3000:
                score += 2
            raw_candidates.append((score, name, arr))
        elif arr.ndim in (1, 2):
            flat = arr.ravel()
            unique = set(np.unique(flat).astype(str).tolist()) if flat.size <= 200 else set()
            score = 0
            if "label" in lname or "class" in lname or lname.split(".")[-1] in {"y", "labels"}:
                score += 10
            if flat.size in (39, 40):
                score += 3
            if unique and unique.issubset({"0", "1", "2"}):
                score += 2
            label_candidates.append((score, name, arr))
    if not raw_candidates or not label_candidates:
        raise KeyError(f"Could not locate 3D raw data and labels in {path}")
    _, raw_name, raw_arr = sorted(raw_candidates, key=lambda x: x[0], reverse=True)[0]
    _, label_name, label_arr = sorted(label_candidates, key=lambda x: x[0], reverse=True)[0]
    rawdata = _normalize_rawdata_shape(raw_arr)
    labels = np.asarray(label_arr).astype(int).ravel()
    if len(labels) != rawdata.shape[0]:
        raise ValueError(f"Label count mismatch: {len(labels)} vs {rawdata.shape}")
    return rawdata, labels, raw_name, label_name

# ---------------------------------------------------------------------------
# 5. Preprocessing (identical to original)
# ---------------------------------------------------------------------------

def make_liu_info(sfreq):
    info = mne.create_info(
        ch_names=EEG_CHANNEL_NAMES,
        sfreq=float(sfreq),
        ch_types=["eeg"] * len(EEG_CHANNEL_NAMES),
    )
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, match_case=False, on_missing="ignore")
    except Exception:
        pass
    return info

def labels_to_zero_based(labels):
    labels = np.asarray(labels).astype(int).ravel()
    unique = set(np.unique(labels).tolist())
    if unique.issubset({1, 2}):
        return labels - 1
    if unique.issubset({0, 1}):
        return labels
    raise ValueError(f"Unexpected labels: {sorted(unique)}")

def preprocess_subject_sjepa_style(rawdata, labels, subject_id):
    if rawdata.ndim != 3:
        raise ValueError(f"Subject {subject_id}: expected 3D rawdata, got {rawdata.shape}")
    n_trials = rawdata.shape[0]
    X_eeg = rawdata[:, EEG_CHANNEL_INDICES, :].astype(np.float64)
    X_eeg *= float(CONFIG.get("source_data_scale", 1.0))
    continuous = X_eeg.transpose(1, 0, 2).reshape(len(EEG_CHANNEL_INDICES), -1)
    info = make_liu_info(CONFIG["source_sfreq"])
    raw = mne.io.RawArray(continuous, info, verbose=False)
    if CONFIG["average_reference_before_resample_filter"]:
        raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.resample(float(CONFIG["sfreq"]), verbose=False)
    raw.filter(float(CONFIG["bandpass_low"]), float(CONFIG["bandpass_high"]), verbose=False)
    if not CONFIG["average_reference_before_resample_filter"]:
        raw.set_eeg_reference("average", projection=False, verbose=False)
    data = raw.get_data()
    expected_samples_per_trial = int(round(rawdata.shape[2] * float(CONFIG["sfreq"]) / float(CONFIG["source_sfreq"])))
    total_expected = n_trials * expected_samples_per_trial
    if data.shape[1] != total_expected:
        n_full = data.shape[1] // n_trials
        expected_samples_per_trial = n_full
        data = data[:, :n_trials * expected_samples_per_trial]
    X_rs = data.reshape(len(EEG_CHANNEL_INDICES), n_trials, expected_samples_per_trial).transpose(1, 0, 2)
    if MI_WINDOW_STOP_SAMPLE > X_rs.shape[-1]:
        raise ValueError(f"Subject {subject_id}: crop exceeds trial length {X_rs.shape[-1]}")
    X_win = X_rs[:, :, MI_WINDOW_START_SAMPLE:MI_WINDOW_STOP_SAMPLE]
    y = labels_to_zero_based(labels)
    return X_win.astype(np.float32), y.astype(np.int64), int(expected_samples_per_trial)

# ---------------------------------------------------------------------------
# 6. Dataset classes
# ---------------------------------------------------------------------------

class SubjectArrayDataset(Dataset):
    def __init__(self, X, y, subject_id):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.subject_id = str(subject_id)

    def __len__(self):
        return int(len(self.y))

    def __getitem__(self, idx):
        return self.X[idx], int(self.y[idx])


class NoisyDataset(Dataset):
    """Wraps a dataset and injects Gaussian noise proportional to per-window std."""

    def __init__(self, dataset, noise_fraction: float = 0.05):
        self.dataset = dataset
        self.noise_fraction = noise_fraction

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = np.asarray(x, dtype=np.float32)
        if self.noise_fraction > 0:
            sigma = self.noise_fraction * float(np.std(x) + 1e-8)
            noise = np.random.randn(*x.shape).astype(np.float32) * sigma
            x = x + noise
        return x, int(y)

# ---------------------------------------------------------------------------
# 7. Locate and load data
# ---------------------------------------------------------------------------

MAT_FILES = []
SOURCE_EXTRACT_DIR = None
for cand in candidate_source_dirs():
    files = find_source_mat_files(cand)
    if files:
        MAT_FILES = files
        SOURCE_EXTRACT_DIR = cand
        break

if not MAT_FILES:
    raise FileNotFoundError(
        "Could not find Liu2024 source .mat files. "
        "Set CONFIG['manual_source_extract_dir'] to your Figshare sourcedata directory."
    )

print(f"Source extract dir: {SOURCE_EXTRACT_DIR}")
print(f"Found {len(MAT_FILES)} .mat files")

subjects = []
for p in MAT_FILES:
    sid = subject_id_from_path(p)
    if CONFIG["subjects_to_use"] is not None and sid not in set(int(s) for s in CONFIG["subjects_to_use"]):
        continue
    if sid in set(int(s) for s in CONFIG["exclude_subjects"]):
        continue
    X_raw, y_raw, raw_field, label_field = load_subject_mat(p)
    subjects.append({
        "subject_id": sid,
        "path": str(p),
        "rawdata_shape": tuple(X_raw.shape),
        "labels_shape": tuple(y_raw.shape),
        "label_counts_raw": np.bincount(y_raw.astype(int), minlength=3).tolist(),
        "raw_field": raw_field,
        "label_field": label_field,
    })

subjects_df = pd.DataFrame(subjects).sort_values("subject_id").reset_index(drop=True)
if subjects_df.empty:
    raise RuntimeError("No subjects loaded.")

SUBJECTS = [int(s) for s in subjects_df["subject_id"].tolist()]
print(f"Subjects loaded: {SUBJECTS}")

# Preprocess all subjects
EEG_INFO = make_liu_info(CONFIG["sfreq"])
CHS_INFO = EEG_INFO["chs"]
CH_NAMES = list(EEG_CHANNEL_NAMES)

Xs, ys, subject_ids = [], [], []
for item in subjects_df.to_dict("records"):
    sid = int(item["subject_id"])
    X_raw, y_raw, _, _ = load_subject_mat(Path(item["path"]))
    X_win, y, _ = preprocess_subject_sjepa_style(X_raw, y_raw, sid)
    Xs.append(X_win)
    ys.append(y)
    subject_ids.extend([sid] * len(y))

X_ALL = np.concatenate(Xs, axis=0)
Y_ALL = np.concatenate(ys, axis=0)
SUBJECT_ID_ALL = np.asarray(subject_ids)
print(f"X_ALL shape: {X_ALL.shape} | Y_ALL counts: {np.bincount(Y_ALL).tolist()}")

def _sort_subject_key(x):
    sx = str(x)
    return int(sx) if sx.isdigit() else sx

SUBJECT_WINDOWS = {}
for sid in sorted(np.unique(SUBJECT_ID_ALL), key=_sort_subject_key):
    idx = np.where(SUBJECT_ID_ALL == sid)[0]
    SUBJECT_WINDOWS[str(sid)] = SubjectArrayDataset(X_ALL[idx], Y_ALL[idx], subject_id=sid)

# ---------------------------------------------------------------------------
# 8. Model building
# ---------------------------------------------------------------------------

NEW_LAYER_PREFIXES = ("spatial_conv.", "final_layer.")

def build_model():
    common_kwargs = {
        "n_chans": len(CH_NAMES),
        "chs_info": CHS_INFO,
        "n_times": WINDOW_SAMPLES,
        "n_outputs": TARGET_N_CLASSES,
    }
    mode = CONFIG["pretrained_mode"]
    if mode == "from_pretrained":
        model = SignalJEPA_PreLocal.from_pretrained(CONFIG["pretrained_repo_id"], **common_kwargs, strict=False)
        info = {"loading_path": "from_pretrained", "repo_id": CONFIG["pretrained_repo_id"], "mode": mode}
    elif mode == "random":
        model = SignalJEPA_PreLocal(**common_kwargs)
        info = {"loading_path": "random_initialization", "repo_id": None, "mode": mode}
    else:
        raise ValueError("pretrained_mode must be 'from_pretrained' or 'random'.")
    info["model_name"] = "SignalJEPA_PreLocal"
    return model, info

def set_trainable_params_for_phase(model, phase):
    if phase not in ("new", "warmup", "full"):
        raise ValueError(f"Unsupported phase: {phase}")
    if phase == "full":
        for _, p in model.named_parameters():
            p.requires_grad = True
        trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
        phase_groups = ["all_parameters"]
    else:
        for _, p in model.named_parameters():
            p.requires_grad = False
        trainable_names = []
        for name, p in model.named_parameters():
            if any(name.startswith(pr) for pr in NEW_LAYER_PREFIXES):
                p.requires_grad = True
                trainable_names.append(name)
        phase_groups = list(NEW_LAYER_PREFIXES)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError(f"No trainable parameters for phase={phase}.")
    return {
        "phase": phase,
        "trainable_groups": phase_groups,
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_ratio": float(trainable / total),
        "trainable_names": trainable_names,
    }

# ---------------------------------------------------------------------------
# 9. Spatial conv weight helpers (identical to original)
# ---------------------------------------------------------------------------

def _json_safe_float(value, decimals=8):
    if value is None:
        return None
    return round(float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)), decimals)

def _json_safe_float_list(values, decimals=8):
    arr = np.asarray(values, dtype=float)
    return np.round(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), decimals=decimals).tolist()

def get_model_module(model_or_clf):
    return model_or_clf.module_ if hasattr(model_or_clf, "module_") else model_or_clf

def _find_first_spatial_weight(model_or_clf):
    model = get_model_module(model_or_clf)
    if not hasattr(model, "spatial_conv"):
        return None, None
    candidates = [
        (name, param.detach().cpu().clone())
        for name, param in model.named_parameters()
        if name.startswith("spatial_conv.") and name.endswith("weight") and param.ndim >= 2
    ]
    if candidates:
        candidates.sort(key=lambda x: (0 if "spatial_conv.1.weight" in x[0] else 1, x[0]))
        return candidates[0]
    for name, module in model.spatial_conv.named_modules():
        if hasattr(module, "weight") and module.weight is not None and module.weight.ndim >= 2:
            return f"spatial_conv.{name}.weight", module.weight.detach().cpu().clone()
    return None, None

def _spatial_weight_to_channel_matrix(weight_tensor, n_chans):
    w = weight_tensor.detach().cpu().float().numpy()
    mat = w.reshape(w.shape[0], -1) if w.ndim > 2 else w.copy()
    if mat.shape[1] == n_chans:
        pass
    elif mat.shape[0] == n_chans:
        mat = mat.T
    elif mat.shape[1] % n_chans == 0:
        mat = mat.reshape(mat.shape[0], -1, n_chans).mean(axis=1)
    elif mat.size % n_chans == 0:
        mat = mat.reshape(-1, n_chans)
    else:
        raise ValueError(f"Cannot reshape spatial weight shape={w.shape} into n_chans={n_chans}.")
    return mat

def get_spatial_conv_weight_matrix(model_or_clf, ch_names):
    param_name, weight_tensor = _find_first_spatial_weight(model_or_clf)
    if weight_tensor is None:
        raise RuntimeError("Could not find spatial_conv weight.")
    return _spatial_weight_to_channel_matrix(weight_tensor, len(ch_names)).copy(), param_name

def compute_spatial_update_stats(initial_weight_matrix, final_weight_matrix, parameter_name):
    if initial_weight_matrix is None or final_weight_matrix is None:
        return None
    w0 = np.asarray(initial_weight_matrix, dtype=float)
    w1 = np.asarray(final_weight_matrix, dtype=float)
    if w0.shape != w1.shape:
        return {"available": False, "reason": f"shape mismatch {w0.shape} vs {w1.shape}"}
    delta = w1 - w0
    init_l2 = float(np.linalg.norm(w0))
    return {
        "available": True,
        "parameter_name": str(parameter_name),
        "shape": list(w1.shape),
        "init_l2": _json_safe_float(init_l2),
        "final_l2": _json_safe_float(np.linalg.norm(w1)),
        "delta_l2": _json_safe_float(np.linalg.norm(delta)),
        "delta_max_abs": _json_safe_float(np.max(np.abs(delta))),
        "relative_delta": _json_safe_float(np.linalg.norm(delta) / max(init_l2, 1e-12)),
        "changed": bool(np.linalg.norm(delta) > 1e-10),
    }

def extract_spatial_conv_summary(model_or_clf, ch_names):
    try:
        weight_matrix, param_name = get_spatial_conv_weight_matrix(model_or_clf, ch_names)
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    abs_mean = np.mean(np.abs(weight_matrix), axis=0)
    signed_mean = np.mean(weight_matrix, axis=0)
    l2_scores = np.linalg.norm(weight_matrix, axis=0)
    return {
        "available": True,
        "parameter_name": param_name,
        "weight_shape": list(weight_matrix.shape),
        "channel_names": list(ch_names),
        "channel_abs_mean": _json_safe_float_list(abs_mean),
        "channel_signed_mean": _json_safe_float_list(signed_mean),
        "channel_l2": _json_safe_float_list(l2_scores),
        "global_abs_mean": _json_safe_float(np.mean(np.abs(weight_matrix))),
        "global_l2": _json_safe_float(np.linalg.norm(weight_matrix)),
    }

def summarize_spatial_update_for_log(stats):
    if not stats or not stats.get("available"):
        return f"Spatial update unavailable: {stats.get('reason') if stats else 'None'}"
    return (
        f"Spatial update | delta_l2={stats['delta_l2']} | "
        f"relative_delta={stats['relative_delta']} | changed={stats['changed']}"
    )

# ---------------------------------------------------------------------------
# 10. Classifier builder
# ---------------------------------------------------------------------------

def get_targets(dataset):
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))], dtype=np.int64)

def make_train_split():
    val_split = CONFIG["val_split"]
    if val_split is None or float(val_split) <= 0.0:
        return None
    return ValidSplit(cv=float(val_split), stratified=True, random_state=12)

def make_callbacks(max_epochs):
    cbs = []
    train_split = make_train_split()
    patience = CONFIG["early_stopping_patience"]
    if train_split is not None and patience is not None and int(patience) > 0:
        cbs.append((
            "early_stopping",
            EarlyStopping(monitor="valid_loss", patience=int(patience), lower_is_better=True, load_best=True),
        ))
    if CONFIG["use_lr_scheduler"]:
        cbs.append((
            "lr_scheduler",
            LRScheduler(
                policy=CosineAnnealingLR,
                T_max=max(max_epochs, 1),
                eta_min=float(CONFIG["lr_min"]),
            ),
        ))
    return cbs

def build_classifier(model, callbacks, max_epochs, fold_seed=None, warm_start=False):
    gen = None
    if fold_seed is not None:
        gen = torch.Generator()
        gen.manual_seed(fold_seed)
    clf_kwargs = {
        "batch_size": CONFIG["batch_size"],
        "max_epochs": int(max_epochs),
        "device": DEVICE,
        "callbacks": callbacks,
        "train_split": make_train_split(),
        "classes": range(TARGET_N_CLASSES),
        "iterator_train__shuffle": True,
        "iterator_train__num_workers": 0,
        "iterator_valid__num_workers": 0,
        "optimizer": torch.optim.AdamW,
        "optimizer__weight_decay": float(CONFIG["optimizer_weight_decay"]),
        "warm_start": warm_start,
    }
    if CONFIG["learning_rate"] is not None:
        clf_kwargs["lr"] = CONFIG["learning_rate"]
    if gen is not None:
        clf_kwargs["iterator_train__generator"] = gen
    return EEGClassifier(model, **clf_kwargs)

def compute_classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

def compute_collapse_diagnostics(y_pred, n_classes):
    pred_hist = np.bincount(np.asarray(y_pred, dtype=int), minlength=n_classes)
    n_pred = int(pred_hist.sum())
    collapse_ratio = float(pred_hist.max() / n_pred) if n_pred else 0.0
    threshold = float(CONFIG.get("collapse_threshold", 0.90))
    return {
        "prediction_histogram": pred_hist.tolist(),
        "collapse_ratio": _json_safe_float(collapse_ratio),
        "collapse_threshold": threshold,
        "collapse_flag": bool(collapse_ratio >= threshold),
        "majority_predicted_class": int(pred_hist.argmax()) if n_pred else None,
    }

# ---------------------------------------------------------------------------
# 11. Training loop
# ---------------------------------------------------------------------------

PRETRAINED_CHECKPOINT_INFO = {}

def run_training_and_eval(train_set, test_set, fold_id, fold_label, n_total_folds=None):
    global PRETRAINED_CHECKPOINT_INFO

    if CONFIG["set_seed"]:
        seed_everything(BASE_SEED)

    y_train = get_targets(train_set)
    y_test = get_targets(test_set)

    strategy = CONFIG["strategy"]
    warmup_epochs = int(CONFIG["warmup_epochs"])
    n_epochs = int(CONFIG["n_epochs"])
    model, pretrained_info = build_model()
    PRETRAINED_CHECKPOINT_INFO = dict(pretrained_info)

    fold_tag = f"/{n_total_folds}" if n_total_folds is not None else ""
    print(f"\nFold {fold_id}{fold_tag} | {fold_label}")
    print(f"    Train: {len(train_set)} | Test: {len(test_set)} | Strategy: {strategy}")

    # Optional noise augmentation on training set
    noise_frac = float(CONFIG.get("augmentation_noise_fraction", 0.0))
    aug_train_set = NoisyDataset(train_set, noise_fraction=noise_frac) if noise_frac > 0 else train_set
    aug_y_train = get_targets(aug_train_set)

    # Snapshot initial spatial conv weights
    initial_spatial_weight_matrix, initial_spatial_param_name = None, None
    if CONFIG["extract_spatial_conv_weights"]:
        try:
            initial_spatial_weight_matrix, initial_spatial_param_name = get_spatial_conv_weight_matrix(model, CH_NAMES)
        except Exception as exc:
            print(f"    WARNING: Could not snapshot initial spatial_conv weights: {exc}")

    if strategy == "new":
        phase_1_summary = set_trainable_params_for_phase(model, "new")
        clf = build_classifier(model, callbacks=make_callbacks(n_epochs), max_epochs=n_epochs, warm_start=False)
        phase_summaries = {"phase_1": phase_1_summary, "phase_2": None}
        clf.fit(aug_train_set, y=aug_y_train)

    elif strategy == "full":
        if warmup_epochs < 1:
            raise ValueError("warmup_epochs must be >= 1 for strategy='full'.")
        # Phase 1: warmup (new layers only)
        phase_1_summary = set_trainable_params_for_phase(model, "warmup")
        print(f"    Phase 1 (warmup): {phase_1_summary['trainable_params']:,}/{phase_1_summary['total_params']:,} params")
        clf = build_classifier(model, callbacks=[], max_epochs=warmup_epochs, warm_start=True)
        clf.fit(aug_train_set, y=aug_y_train)

        # Phase 2: full fine-tuning
        remaining = n_epochs - warmup_epochs
        if remaining < 1:
            raise ValueError("n_epochs must be > warmup_epochs for strategy='full'.")
        phase_2_summary = set_trainable_params_for_phase(clf.module_, "full")
        print(f"    Phase 2 (full): {phase_2_summary['trainable_params']:,}/{phase_2_summary['total_params']:,} params")
        clf.initialize_optimizer()
        clf.set_params(callbacks=make_callbacks(remaining), max_epochs=remaining)
        clf.fit(aug_train_set, y=aug_y_train)
        phase_summaries = {"phase_1": phase_1_summary, "phase_2": phase_2_summary}
    else:
        raise ValueError("CONFIG['strategy'] must be 'new' or 'full'.")

    y_pred = clf.predict(test_set)
    metrics = compute_classification_metrics(y_test, y_pred)
    collapse = compute_collapse_diagnostics(y_pred, TARGET_N_CLASSES)

    # Spatial update stats
    spatial_update_stats = None
    if CONFIG["extract_spatial_conv_weights"] and initial_spatial_weight_matrix is not None:
        try:
            final_w, final_name = get_spatial_conv_weight_matrix(clf.module_, CH_NAMES)
            spatial_update_stats = compute_spatial_update_stats(
                initial_spatial_weight_matrix, final_w,
                final_name or initial_spatial_param_name,
            )
        except Exception as exc:
            spatial_update_stats = {"available": False, "reason": str(exc)}

    spatial_summary = None
    if CONFIG["extract_spatial_conv_weights"]:
        spatial_summary = extract_spatial_conv_summary(clf.module_, CH_NAMES)

    stopped_epoch = int(clf.history[-1]["epoch"]) if clf.history else 0
    valid_loss_curve = [(int(r["epoch"]), float(r["valid_loss"])) for r in clf.history if "valid_loss" in r]
    best_epoch, best_valid_loss = (min(valid_loss_curve, key=lambda x: x[1]) if valid_loss_curve else (None, None))
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(TARGET_N_CLASSES)).tolist()

    print(
        f"    Result | best_epoch={best_epoch} | stop={stopped_epoch} | "
        f"acc={metrics['accuracy']:.4f} | bal_acc={metrics['balanced_accuracy']:.4f} | "
        f"collapse={collapse['collapse_flag']}"
    )
    if spatial_update_stats:
        print("    " + summarize_spatial_update_for_log(spatial_update_stats))

    return {
        "fold_id": int(fold_id),
        "fold_label": str(fold_label),
        "model_name": "SignalJEPA_PreLocal",
        "strategy": strategy,
        "n_train": int(len(train_set)),
        "n_test": int(len(test_set)),
        "best_epoch": best_epoch,
        "stopped_epoch": int(stopped_epoch),
        "best_valid_loss": best_valid_loss,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "confusion_matrix": cm,
        "prediction_histogram": collapse["prediction_histogram"],
        "collapse_diagnostics": collapse,
        "spatial_conv": spatial_summary,
        "spatial_update_stats": spatial_update_stats,
    }

def make_fold_splits(y, n_folds, n_classes):
    counts = np.bincount(y, minlength=n_classes)
    if counts.min() < n_folds:
        raise ValueError(f"Cannot use {n_folds} folds with class counts={counts.tolist()}.")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=12)
    indices = np.arange(len(y))
    return [
        {"fold_id": fid, "idx_train": tr, "idx_test": te}
        for fid, (tr, te) in enumerate(skf.split(indices, y), start=1)
    ]

def run_subject_cv(subject_id, subject_dataset, n_classes, cv_folds):
    y = get_targets(subject_dataset)
    counts = np.bincount(y, minlength=n_classes)
    print(f"\nSubject {subject_id}: {len(subject_dataset)} windows | class_counts={counts.tolist()}")
    folds = make_fold_splits(y, n_folds=cv_folds, n_classes=n_classes)
    results = []
    for fold in folds:
        train_set = Subset(subject_dataset, fold["idx_train"].tolist())
        test_set = Subset(subject_dataset, fold["idx_test"].tolist())
        result = run_training_and_eval(
            train_set, test_set, fold["fold_id"], f"subject={subject_id}", n_total_folds=cv_folds
        )
        result["subject_id"] = str(subject_id)
        results.append(result)
    accs = [r["accuracy"] for r in results if r["accuracy"] is not None]
    bals = [r["balanced_accuracy"] for r in results if r["balanced_accuracy"] is not None]
    print(
        f"  Subject {subject_id}: acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
        f"bal_acc={np.mean(bals):.4f}±{np.std(bals):.4f}"
    )
    return results

# ---------------------------------------------------------------------------
# 12. Run within-subject CV
# ---------------------------------------------------------------------------

print("=" * 70)
print("STARTING WITHIN-SUBJECT CV (v2 — full fine-tuning + augmentation)")
print("=" * 70)
print(f"Dataset:    {CONFIG['dataset_name']}")
print(f"Strategy:   {CONFIG['strategy']}  (warmup={CONFIG['warmup_epochs']}, total={CONFIG['n_epochs']} epochs)")
print(f"Optimizer:  AdamW  lr={CONFIG['learning_rate']}  wd={CONFIG['optimizer_weight_decay']}")
print(f"Noise aug:  fraction={CONFIG['augmentation_noise_fraction']}")
print(f"Device:     {DEVICE}")
print("=" * 70)

FOLD_RESULTS = []
sorted_sids = sorted(SUBJECT_WINDOWS.keys(), key=_sort_subject_key)

for sid in sorted_sids:
    subject_results = run_subject_cv(sid, SUBJECT_WINDOWS[sid], TARGET_N_CLASSES, CONFIG["cv_folds"])
    FOLD_RESULTS.extend(subject_results)

# ---------------------------------------------------------------------------
# 13. Aggregate and save results
# ---------------------------------------------------------------------------

def aggregate_results(fold_results):
    grouped = {}
    for r in fold_results:
        sid = r.get("subject_id", "global")
        grouped.setdefault(sid, {"accuracies": [], "balanced_accuracies": []})
        grouped[sid]["accuracies"].append(r.get("accuracy"))
        grouped[sid]["balanced_accuracies"].append(r.get("balanced_accuracy"))
    for sid, m in grouped.items():
        avs = [v for v in m["accuracies"] if v is not None]
        bvs = [v for v in m["balanced_accuracies"] if v is not None]
        m["mean_accuracy"] = float(np.mean(avs)) if avs else None
        m["std_accuracy"] = float(np.std(avs)) if avs else None
        m["mean_balanced_accuracy"] = float(np.mean(bvs)) if bvs else None
        m["std_balanced_accuracy"] = float(np.std(bvs)) if bvs else None
    all_accs = [r["accuracy"] for r in fold_results if r.get("accuracy") is not None]
    all_bals = [r["balanced_accuracy"] for r in fold_results if r.get("balanced_accuracy") is not None]
    global_metrics = {
        "mean_accuracy": float(np.mean(all_accs)) if all_accs else None,
        "std_accuracy": float(np.std(all_accs)) if all_accs else None,
        "mean_balanced_accuracy": float(np.mean(all_bals)) if all_bals else None,
        "std_balanced_accuracy": float(np.std(all_bals)) if all_bals else None,
        "n_subjects": len(grouped),
        "n_folds_total": len(fold_results),
    }
    return grouped, global_metrics

SUBJECT_METRICS, GLOBAL_METRICS = aggregate_results(FOLD_RESULTS)

print("\n" + "=" * 70)
print("AGGREGATED RESULTS (v2)")
print("=" * 70)
for sid, m in sorted(SUBJECT_METRICS.items(), key=lambda x: _sort_subject_key(x[0])):
    acc_str = f"{m['mean_accuracy']:.4f}±{m['std_accuracy']:.4f}" if m["mean_accuracy"] is not None else "N/A"
    bal_str = f"{m['mean_balanced_accuracy']:.4f}±{m['std_balanced_accuracy']:.4f}" if m["mean_balanced_accuracy"] is not None else "N/A"
    print(f"  {sid}: acc={acc_str}  bal_acc={bal_str}")
print("-" * 70)
print(
    f"  OVERALL: acc={GLOBAL_METRICS['mean_accuracy']:.4f}±{GLOBAL_METRICS['std_accuracy']:.4f}  "
    f"bal_acc={GLOBAL_METRICS['mean_balanced_accuracy']:.4f}±{GLOBAL_METRICS['std_balanced_accuracy']:.4f}"
)
print("=" * 70)
print(f"\nBASELINE (original): acc=0.4925±0.1643  bal_acc=0.4925±0.1643")
delta = (GLOBAL_METRICS["mean_accuracy"] or 0) - 0.4925
print(f"Improvement:         Δacc={delta:+.4f}")
print("=" * 70)

# Save artifacts
cv_path = ARTIFACT_DIR / "cv_results.json"
with open(cv_path, "w") as f:
    json.dump(FOLD_RESULTS, f, indent=2)
with open(ARTIFACT_DIR / "subject_metrics.json", "w") as f:
    json.dump(SUBJECT_METRICS, f, indent=2)
with open(ARTIFACT_DIR / "global_metrics.json", "w") as f:
    json.dump(GLOBAL_METRICS, f, indent=2)

print(f"CV results saved to: {cv_path}")
_LOG_FILE_HANDLE.close()
