"""
Liu2024 Source .mat SignalJEPA PreLocal — Leave-One-Subject-Out (LOSO) Cross-Validation

Applies all v2 improvements (full fine-tuning, AdamW, CosineAnnealingLR, early stopping,
noise augmentation) in a LOSO protocol:
  - For each of 50 subjects: train on the remaining 49 subjects (~1960 samples),
    test on the held-out subject (40 samples).
  - No within-subject data leakage; a strict cross-subject evaluation.
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import re
import sys
import json
import hashlib
import random
import builtins
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, ConcatDataset

from scipy.io import loadmat

import mne
mne.set_log_level("WARNING")

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit
from torch.optim.lr_scheduler import CosineAnnealingLR

from braindecode import EEGClassifier
from braindecode.models import SignalJEPA_PreLocal

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

WORKING_DIR = Path(__file__).resolve().parent

CONFIG = {
    # Paths
    "artifact_dir": str(WORKING_DIR / "artifacts" / "liu2024-source-mat-sjepa-prelocal-loso"),
    "manual_source_extract_dir": None,

    # Dataset
    "dataset_name": "Liu2024_SourceMAT_LOSO",
    "labels_to_keep": ["left_hand", "right_hand"],
    "subjects_to_use": None,
    "exclude_subjects": [],

    # Source data conventions
    "source_sfreq": 500,
    "source_trial_duration_s": 8.0,
    "drop_source_reference_channel": True,
    "drop_eog_and_marker": True,
    "source_data_scale": 1.0,

    # S-JEPA preprocessing
    "sfreq": 128,
    "bandpass_low": 0.5,
    "bandpass_high": 40.0,
    "average_reference_before_resample_filter": True,

    # S-JEPA window
    "target_window_samples": 537,
    "mi_window_start_s": 2.0,

    # Model
    "model_name": "SignalJEPA_PreLocal",
    "pretrained_mode": "from_pretrained",
    "pretrained_repo_id": "braindecode/signal-jepa_without-chans",

    # Full fine-tuning with warmup
    "strategy": "full",
    "warmup_epochs": 10,

    # Training hyperparameters (LOSO has much larger train sets, so fewer epochs needed)
    "batch_size": 32,
    "n_epochs": 50,
    "early_stopping_patience": 10,
    "learning_rate": 5e-4,

    # AdamW
    "optimizer_weight_decay": 1e-4,

    # Cosine LR schedule
    "use_lr_scheduler": True,
    "lr_min": 1e-6,

    # Gaussian noise augmentation
    "augmentation_noise_fraction": 0.05,

    # Validation split within training set (to monitor loss for early stopping)
    "val_split": 0.05,  # 5% of ~1960 = ~98 samples for validation

    # Reproducibility
    "seed": 12,
    "set_seed": True,

    # Diagnostics
    "collapse_threshold": 0.90,
    "log_spatial_update_stats": True,
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
# 3. Reproducibility & device
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
# 4. Data loading helpers
# ---------------------------------------------------------------------------

def find_source_mat_files(root):
    root = Path(root)
    return sorted(root.rglob("*.mat")) if root.exists() else []

def candidate_source_dirs():
    candidates = []
    if CONFIG["manual_source_extract_dir"] is not None:
        candidates.append(Path(CONFIG["manual_source_extract_dir"]))

    search_roots = [WORKING_DIR, WORKING_DIR.parent, WORKING_DIR.parent.parent]

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
# 5. Preprocessing
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
# 7. Load and preprocess all subjects
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

subjects_meta = []
for p in MAT_FILES:
    sid = subject_id_from_path(p)
    if CONFIG["subjects_to_use"] is not None and sid not in set(int(s) for s in CONFIG["subjects_to_use"]):
        continue
    if sid in set(int(s) for s in CONFIG["exclude_subjects"]):
        continue
    subjects_meta.append({"subject_id": sid, "path": str(p)})

subjects_df = pd.DataFrame(subjects_meta).sort_values("subject_id").reset_index(drop=True)
if subjects_df.empty:
    raise RuntimeError("No subjects loaded.")

SUBJECTS = [int(s) for s in subjects_df["subject_id"].tolist()]
print(f"Subjects loaded: {SUBJECTS}")

EEG_INFO = make_liu_info(CONFIG["sfreq"])
CHS_INFO = EEG_INFO["chs"]
CH_NAMES = list(EEG_CHANNEL_NAMES)

# Store per-subject preprocessed data
SUBJECT_DATASETS: dict[int, SubjectArrayDataset] = {}
for item in subjects_df.to_dict("records"):
    sid = int(item["subject_id"])
    X_raw, y_raw, _, _ = load_subject_mat(Path(item["path"]))
    X_win, y, _ = preprocess_subject_sjepa_style(X_raw, y_raw, sid)
    SUBJECT_DATASETS[sid] = SubjectArrayDataset(X_win, y, subject_id=sid)

print(f"Preprocessed {len(SUBJECT_DATASETS)} subjects")

# ---------------------------------------------------------------------------
# 8. Model helpers
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
    elif mode == "random":
        model = SignalJEPA_PreLocal(**common_kwargs)
    else:
        raise ValueError("pretrained_mode must be 'from_pretrained' or 'random'.")
    return model

def set_trainable_params_for_phase(model, phase):
    if phase == "full":
        for _, p in model.named_parameters():
            p.requires_grad = True
    else:  # warmup
        for _, p in model.named_parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if any(name.startswith(pr) for pr in NEW_LAYER_PREFIXES):
                p.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def _find_first_spatial_weight(model):
    if not hasattr(model, "spatial_conv"):
        return None, None
    for name, param in model.named_parameters():
        if name.startswith("spatial_conv.") and name.endswith("weight") and param.ndim >= 2:
            return name, param.detach().cpu().clone()
    return None, None

def get_spatial_weight(model):
    _, w = _find_first_spatial_weight(model)
    return w

def spatial_update_stats(w0, w1):
    if w0 is None or w1 is None:
        return None
    delta = (w1 - w0).norm().item()
    init_norm = w0.norm().item()
    return {
        "delta_l2": round(delta, 8),
        "relative_delta": round(delta / max(init_norm, 1e-12), 8),
        "changed": delta > 1e-10,
    }

# ---------------------------------------------------------------------------
# 9. Classifier builder
# ---------------------------------------------------------------------------

def get_targets(dataset):
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))], dtype=np.int64)

def make_callbacks(max_epochs):
    cbs = []
    val_split_val = CONFIG["val_split"]
    train_split = ValidSplit(cv=float(val_split_val), stratified=True, random_state=12) if val_split_val else None
    patience = CONFIG["early_stopping_patience"]
    if train_split is not None and patience is not None and int(patience) > 0:
        cbs.append((
            "early_stopping",
            EarlyStopping(monitor="valid_loss", patience=int(patience), lower_is_better=True, load_best=True),
        ))
    if CONFIG["use_lr_scheduler"]:
        remaining = max(1, max_epochs - CONFIG["warmup_epochs"])
        cbs.append((
            "lr_scheduler",
            LRScheduler(
                policy=CosineAnnealingLR,
                T_max=remaining,
                eta_min=float(CONFIG["lr_min"]),
                event_name="batch_end",
            ),
        ))
    return cbs, train_split

def build_clf(model, max_epochs, train_split, callbacks):
    import torch.optim as optim
    return EEGClassifier(
        module=model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=optim.AdamW,
        optimizer__lr=float(CONFIG["learning_rate"]),
        optimizer__weight_decay=float(CONFIG["optimizer_weight_decay"]),
        max_epochs=max_epochs,
        batch_size=int(CONFIG["batch_size"]),
        train_split=train_split,
        callbacks=callbacks,
        device=str(DEVICE),
        verbose=1,
    )

# ---------------------------------------------------------------------------
# 10. LOSO training loop
# ---------------------------------------------------------------------------

def train_one_phase(clf, dataset, max_epochs, warm=False):
    """Fit clf for max_epochs."""
    clf.max_epochs = max_epochs
    clf.fit(dataset, y=None)

def run_loso():
    all_results = []

    print("=" * 70)
    print("STARTING LOSO CV")
    print("=" * 70)
    print(f"Dataset:    {CONFIG['dataset_name']}")
    print(f"Subjects:   {len(SUBJECTS)}")
    print(f"Strategy:   full  (warmup={CONFIG['warmup_epochs']}, total={CONFIG['n_epochs']} epochs)")
    print(f"Optimizer:  AdamW  lr={CONFIG['learning_rate']}  wd={CONFIG['optimizer_weight_decay']}")
    print(f"Noise aug:  fraction={CONFIG['augmentation_noise_fraction']}")
    print(f"Device:     {DEVICE}")
    print("=" * 70)

    for fold_idx, test_sid in enumerate(SUBJECTS):
        train_sids = [s for s in SUBJECTS if s != test_sid]
        train_datasets = [SUBJECT_DATASETS[s] for s in train_sids]
        train_ds_raw = ConcatDataset(train_datasets)
        train_ds = NoisyDataset(train_ds_raw, noise_fraction=CONFIG["augmentation_noise_fraction"])
        test_ds = SUBJECT_DATASETS[test_sid]

        n_train = len(train_ds)
        n_test = len(test_ds)
        train_labels = np.concatenate([get_targets(SUBJECT_DATASETS[s]) for s in train_sids])
        test_labels = get_targets(test_ds)

        print(f"\nLOSO fold {fold_idx + 1}/{len(SUBJECTS)} | test_subject={test_sid}")
        print(f"    Train: {n_train} | Test: {n_test}")

        # Build fresh model for each fold
        model = build_model()
        w0 = get_spatial_weight(model)

        # ---- Phase 1: warmup (new layers only) ----
        warmup_epochs = CONFIG["warmup_epochs"]
        total_params, trainable_params = set_trainable_params_for_phase(model, "warmup")
        print(f"    Phase 1 (warmup): {trainable_params:,}/{total_params:,} params")

        callbacks_warm, train_split = make_callbacks(warmup_epochs)
        clf = build_clf(model, warmup_epochs, train_split, callbacks_warm)
        clf.fit(train_ds, y=train_labels)

        # ---- Phase 2: full fine-tuning ----
        remaining = CONFIG["n_epochs"] - warmup_epochs
        total_params, trainable_params = set_trainable_params_for_phase(clf.module_, "full")
        print(f"    Phase 2 (full): {trainable_params:,}/{total_params:,} params")

        # Rebuild clf to reset optimizer over all parameters
        callbacks_full, train_split2 = make_callbacks(remaining)
        clf_full = build_clf(clf.module_, remaining, train_split2, callbacks_full)
        clf_full.fit(train_ds, y=train_labels)

        # ---- Evaluate on held-out subject ----
        X_test = test_ds.X
        y_test = test_ds.y
        y_pred = clf_full.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        bal_acc = float(balanced_accuracy_score(y_test, y_pred))

        # Detect collapse
        unique_pred = np.unique(y_pred)
        collapsed = len(unique_pred) == 1 or (np.bincount(y_pred, minlength=2).max() / len(y_pred) >= CONFIG["collapse_threshold"])

        w1 = get_spatial_weight(clf_full.module_)
        stats = spatial_update_stats(w0, w1)

        print(f"    Result | acc={acc:.4f} | bal_acc={bal_acc:.4f} | collapse={collapsed}")
        if stats and CONFIG["log_spatial_update_stats"]:
            print(f"    Spatial update | delta_l2={stats['delta_l2']} | relative_delta={stats['relative_delta']} | changed={stats['changed']}")

        result = {
            "fold": fold_idx + 1,
            "test_subject": int(test_sid),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "acc": acc,
            "bal_acc": bal_acc,
            "collapsed": bool(collapsed),
            "spatial_update": stats,
        }
        all_results.append(result)

    return all_results

# ---------------------------------------------------------------------------
# 11. Main
# ---------------------------------------------------------------------------

loso_results = run_loso()

accs = np.array([r["acc"] for r in loso_results])
bal_accs = np.array([r["bal_acc"] for r in loso_results])

print("\n" + "=" * 70)
print("LOSO RESULTS PER SUBJECT")
print("=" * 70)
for r in loso_results:
    print(f"  sub-{r['test_subject']:02d}: acc={r['acc']:.4f} | bal_acc={r['bal_acc']:.4f} | collapsed={r['collapsed']}")

print(f"\n  OVERALL: acc={accs.mean():.4f}±{accs.std():.4f}  bal_acc={bal_accs.mean():.4f}±{bal_accs.std():.4f}")
print(f"  Chance level: 0.5000 (balanced binary)")
print("=" * 70)

results_path = ARTIFACT_DIR / "loso_results.json"
with open(results_path, "w") as f:
    json.dump({
        "run_id": RUN_ID,
        "config": CONFIG,
        "overall": {
            "mean_acc": float(accs.mean()),
            "std_acc": float(accs.std()),
            "mean_bal_acc": float(bal_accs.mean()),
            "std_bal_acc": float(bal_accs.std()),
        },
        "per_subject": loso_results,
    }, f, indent=2)

print(f"\nLOSO results saved to: {results_path}")

_LOG_FILE_HANDLE.flush()
_LOG_FILE_HANDLE.close()
