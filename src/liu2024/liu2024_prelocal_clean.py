"""Leakage-safe Liu2024 preprocessing and SignalJEPA PreLocal evaluation helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import mne
import numpy as np
import torch
from braindecode.models import SignalJEPA_PreLocal
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset


LIU_SOURCE_SFREQ = 500
LIU_TRIALS = 40
LIU_CHANNELS = 33
LIU_SAMPLES = 4000
LIU_EEG_INDICES = [i for i in range(30) if i != 17]
LIU_EEG_NAMES = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4",
    "FT7", "FT8", "Cz", "C3", "C4", "T3", "T4", "CP3", "CP4", "TP7",
    "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2",
]
MARKER_INDEX = 32


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def subject_id_from_path(path: Path) -> int:
    name = path.parent.name
    if not name.startswith("sub-"):
        raise ValueError(f"Expected sub-XX parent for {path}")
    return int(name.split("-")[-1])


def make_info(sfreq: float) -> mne.Info:
    info = mne.create_info(LIU_EEG_NAMES, sfreq, ch_types="eeg")
    info.set_montage(
        mne.channels.make_standard_montage("standard_1020"),
        match_case=False,
        on_missing="raise",
    )
    return info


def load_subject(path: Path) -> dict:
    """Load the documented Liu MATLAB fields and fail closed on layout drift."""
    mat = loadmat(path)
    if "eeg" not in mat:
        raise KeyError(f"Missing top-level eeg struct in {path}")
    eeg = mat["eeg"][0, 0]
    names = eeg.dtype.names or ()
    if "rawdata" not in names or "label" not in names:
        raise KeyError(f"Missing eeg.rawdata/eeg.label in {path}; fields={names}")
    raw = np.asarray(eeg["rawdata"], dtype=np.float64)
    labels_raw = np.asarray(eeg["label"]).reshape(-1).astype(np.int64)
    if raw.shape != (LIU_TRIALS, LIU_CHANNELS, LIU_SAMPLES):
        raise ValueError(f"Unexpected raw shape in {path}: {raw.shape}")
    if labels_raw.shape != (LIU_TRIALS,) or set(np.unique(labels_raw)) != {1, 2}:
        raise ValueError(f"Unexpected labels in {path}: shape={labels_raw.shape}, values={np.unique(labels_raw)}")
    labels = labels_raw - 1
    if np.bincount(labels, minlength=2).tolist() != [20, 20]:
        raise ValueError(f"Unbalanced labels in {path}")

    onsets, ends = [], []
    marker = raw[:, MARKER_INDEX]
    for trial_index, trial_marker in enumerate(marker):
        onset_hits = np.flatnonzero(trial_marker == 2)
        end_hits = np.flatnonzero(trial_marker == 3)
        if len(onset_hits) != 1 or len(end_hits) != 1:
            raise ValueError(
                f"{path} trial {trial_index}: expected one marker 2 and one marker 3, "
                f"got {len(onset_hits)} and {len(end_hits)}"
            )
        onset, end = int(onset_hits[0]), int(end_hits[0])
        if not 2000 <= end - onset <= 2003 or onset + 2000 > LIU_SAMPLES:
            raise ValueError(f"{path} trial {trial_index}: invalid marker interval {onset}:{end}")
        onsets.append(onset)
        ends.append(end)
    return {
        "subject_id": subject_id_from_path(path),
        "path": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "raw": raw,
        "labels": labels,
        "onsets": np.asarray(onsets, dtype=np.int64),
        "ends": np.asarray(ends, dtype=np.int64),
    }


def preprocess_subject(subject: dict, config: dict) -> tuple[np.ndarray, list[dict]]:
    """Process each complete trial independently, then take marker-relative MI."""
    target_sfreq = int(config["target_sfreq"])
    target_samples = int(round(float(config["mi_window_seconds"]) * target_sfreq))
    outputs, records = [], []
    source_info = make_info(LIU_SOURCE_SFREQ)
    for trial_index in range(LIU_TRIALS):
        eeg_uv = subject["raw"][trial_index, LIU_EEG_INDICES]
        raw = mne.io.RawArray(eeg_uv * 1e-6, source_info.copy(), verbose=False)
        if config.get("average_reference", True):
            raw.set_eeg_reference("average", projection=False, verbose=False)
        band = config.get("bandpass_hz")
        if band is not None:
            raw.filter(
                l_freq=float(band[0]),
                h_freq=float(band[1]),
                method="fir",
                phase="zero",
                fir_design="firwin",
                verbose=False,
            )
        if target_sfreq != LIU_SOURCE_SFREQ:
            raw.resample(target_sfreq, verbose=False)
        full = raw.get_data() * 1e6
        onset_source = int(subject["onsets"][trial_index])
        onset_target = int(round(onset_source * target_sfreq / LIU_SOURCE_SFREQ))
        stop_target = onset_target + target_samples
        if stop_target > full.shape[-1]:
            raise ValueError(
                f"Subject {subject['subject_id']} trial {trial_index}: marker crop "
                f"[{onset_target}:{stop_target}] exceeds {full.shape[-1]} samples"
            )
        window = full[:, onset_target:stop_target]
        if window.shape != (len(LIU_EEG_INDICES), target_samples) or not np.isfinite(window).all():
            raise ValueError(f"Invalid preprocessed window: {window.shape}")
        outputs.append(window.astype(np.float32))
        records.append({
            "subject_id": int(subject["subject_id"]),
            "trial_index": trial_index,
            "label": int(subject["labels"][trial_index]),
            "marker2_source_sample": onset_source,
            "marker3_source_sample": int(subject["ends"][trial_index]),
            "marker_duration_samples": int(subject["ends"][trial_index] - onset_source),
            "marker2_target_sample": onset_target,
            "crop_stop_target_sample": stop_target,
            "source_path": subject["path"],
        })
    return np.stack(outputs), records


def make_outer_splits(labels: np.ndarray, config: dict) -> list[dict]:
    splitter = StratifiedKFold(
        n_splits=int(config["cv_folds"]),
        shuffle=True,
        random_state=int(config["cv_seed"]),
    )
    splits = []
    seen = np.zeros(len(labels), dtype=np.int64)
    for fold_id, (train, test) in enumerate(splitter.split(np.arange(len(labels)), labels), 1):
        if set(train) & set(test):
            raise AssertionError("Outer train/test overlap")
        seen[test] += 1
        splits.append({"fold_id": fold_id, "train_indices": train, "test_indices": test})
    if not np.all(seen == 1):
        raise AssertionError(f"Expected exact-once outer testing, got counts={seen.tolist()}")
    return splits


def make_inner_split(labels: np.ndarray, outer_train: np.ndarray, config: dict, fold_id: int) -> tuple[np.ndarray, np.ndarray]:
    local_labels = labels[outer_train]
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(config["val_fraction"]),
        random_state=int(config["val_seed"]) + fold_id,
    )
    train_local, valid_local = next(splitter.split(np.arange(len(outer_train)), local_labels))
    inner_train, valid = outer_train[train_local], outer_train[valid_local]
    if set(inner_train) & set(valid):
        raise AssertionError("Inner train/validation overlap")
    return inner_train, valid


def fit_normalizer(x: np.ndarray, mode: str, eps: float) -> dict:
    mode = str(mode).lower()
    if mode == "none":
        return {"mode": mode, "hash": None}
    if mode != "train_channel_zscore":
        raise ValueError(f"Unsupported clean normalizer: {mode}")
    mean = x.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    scale = x.std(axis=(0, 2), keepdims=True).astype(np.float32)
    scale = np.maximum(scale, eps)
    return {"mode": mode, "mean": mean, "scale": scale, "hash": array_hash(mean, scale)}


def apply_normalizer(x: np.ndarray, state: dict) -> np.ndarray:
    if state["mode"] == "none":
        return x.astype(np.float32)
    return ((x - state["mean"]) / state["scale"]).astype(np.float32)


def validate_pretraining_export(config: dict, n_times: int) -> tuple[dict, dict]:
    """Validate a completed local SSL export before any downstream fitting."""
    checkpoint_path = config.get("pretrained_checkpoint_path")
    if not checkpoint_path:
        raise RuntimeError("A local pretrained_checkpoint_path is required")
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if config.get("require_best_checkpoint_filename", True) and checkpoint_path.name != "student_backbone_best.pt":
        raise RuntimeError(f"Expected student_backbone_best.pt, got {checkpoint_path.name}")

    expected_sha256 = config.get("pretrained_checkpoint_sha256")
    if config.get("require_checkpoint_sha256", True) and not expected_sha256:
        raise RuntimeError("pretrained_checkpoint_sha256 is required for the locked evaluation")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if expected_sha256 and checkpoint_sha256 != str(expected_sha256).lower():
        raise RuntimeError(f"Checkpoint SHA256 mismatch: expected {expected_sha256}, got {checkpoint_sha256}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "student_backbone_state_dict" not in payload:
        raise RuntimeError("Local checkpoint must contain student_backbone_state_dict")
    epoch = int(payload.get("epoch", -1))
    if epoch <= 0 and not config.get("allow_untrained_checkpoint_for_smoke", False):
        raise RuntimeError(f"Refusing untrained/verification export with epoch={epoch}")

    split = payload.get("subject_split")
    if not isinstance(split, dict):
        raise RuntimeError("Checkpoint is missing subject_split provenance")
    observed_split = {
        "train_subject_ids": sorted(int(x) for x in split.get("train_subject_ids", [])),
        "val_subject_ids": sorted(int(x) for x in split.get("val_subject_ids", [])),
        "excluded_subject_ids": sorted(int(x) for x in split.get("excluded_subject_ids", [])),
    }
    expected_split = {
        "train_subject_ids": sorted(int(x) for x in config["expected_ssl_train_subject_ids"]),
        "val_subject_ids": sorted(int(x) for x in config["expected_ssl_val_subject_ids"]),
        "excluded_subject_ids": sorted(int(x) for x in config["expected_ssl_excluded_subject_ids"]),
    }
    if observed_split != expected_split:
        raise RuntimeError(f"Checkpoint SSL split mismatch: observed={observed_split}, expected={expected_split}")
    split_sets = [set(observed_split[key]) for key in observed_split]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Checkpoint SSL subject sets are not pairwise disjoint")
    if set().union(*split_sets) != set(range(1, 51)) and not config.get("allow_incomplete_split_for_smoke", False):
        raise RuntimeError("Checkpoint SSL subject sets do not cover exactly Liu2024 subjects 1-50")

    expected_input_seconds = n_times / float(config["target_sfreq"])
    if int(payload.get("sfreq", -1)) != int(config["target_sfreq"]):
        raise RuntimeError(f"Checkpoint sampling rate mismatch: {payload.get('sfreq')}")
    if not np.isclose(float(payload.get("input_window_seconds", -1)), expected_input_seconds):
        raise RuntimeError(f"Checkpoint input duration mismatch: {payload.get('input_window_seconds')}")
    if list(payload.get("ch_names", [])) != LIU_EEG_NAMES:
        raise RuntimeError("Checkpoint channel names/order do not match canonical Liu29")

    preprocessing = payload.get("preprocessing_config")
    if not isinstance(preprocessing, dict):
        raise RuntimeError("Checkpoint is missing preprocessing_config provenance")
    expected_preprocessing = {
        "sfreq": int(config["target_sfreq"]),
        "bandpass_low": float(config["bandpass_hz"][0]),
        "bandpass_high": float(config["bandpass_hz"][1]),
        "pretrain_duration_s": float(config["mi_window_seconds"]),
        "window_size_samples": int(n_times),
        "filter_method": "fir",
        "model_input_unit": "microvolts",
    }
    mismatches = {
        key: {"observed": preprocessing.get(key), "expected": expected}
        for key, expected in expected_preprocessing.items()
        if preprocessing.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint preprocessing mismatch: {mismatches}")

    source_state = payload["student_backbone_state_dict"]
    if not isinstance(source_state, dict) or not source_state:
        raise RuntimeError("student_backbone_state_dict is empty or invalid")
    nonfinite = [name for name, value in source_state.items() if not torch.is_tensor(value) or not torch.isfinite(value).all()]
    if nonfinite:
        raise RuntimeError(f"Checkpoint has invalid/non-finite tensors: {nonfinite[:10]}")
    audit = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": epoch,
        "subject_split": observed_split,
        "preprocessing_config": preprocessing,
        "ch_names": list(payload["ch_names"]),
        "all_state_keys": sorted(source_state),
    }
    return payload, audit


def build_model(config: dict, n_times: int) -> tuple[torch.nn.Module, dict]:
    kwargs = {
        "n_chans": len(LIU_EEG_NAMES),
        "chs_info": make_info(config["target_sfreq"])["chs"],
        "n_times": n_times,
        "n_outputs": 2,
    }
    checkpoint_path = config.get("pretrained_checkpoint_path")
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path).resolve()
        export_audit = None
        if config.get("require_validated_pretraining_export", False):
            payload, export_audit = validate_pretraining_export(config, n_times)
            checkpoint_sha256 = export_audit["checkpoint_sha256"]
        else:
            checkpoint_sha256 = sha256_file(checkpoint_path)
            expected_sha256 = config.get("pretrained_checkpoint_sha256")
            if expected_sha256 and checkpoint_sha256 != expected_sha256:
                raise RuntimeError(
                    f"Checkpoint SHA256 mismatch: expected {expected_sha256}, got {checkpoint_sha256}"
                )
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = SignalJEPA_PreLocal(**kwargs)
        if not isinstance(payload, dict) or "student_backbone_state_dict" not in payload:
            raise RuntimeError("Local checkpoint must contain student_backbone_state_dict")
        source_state = payload["student_backbone_state_dict"]
        target_state = model.state_dict()
        expected_keys = {key for key in target_state if key.startswith("feature_encoder.")}
        loaded_state = {key: value for key, value in source_state.items() if key in expected_keys}
        if set(loaded_state) != expected_keys:
            missing = sorted(expected_keys - set(loaded_state))
            raise RuntimeError(f"Local checkpoint does not cover the full PreLocal feature encoder: {missing}")
        shape_errors = {
            key: (tuple(loaded_state[key].shape), tuple(target_state[key].shape))
            for key in expected_keys
            if loaded_state[key].shape != target_state[key].shape
        }
        if shape_errors:
            raise RuntimeError(f"Local checkpoint feature-encoder shape mismatch: {shape_errors}")
        incompatible = model.load_state_dict(loaded_state, strict=False)
        expected_missing = set(target_state) - expected_keys
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected local checkpoint load result: {incompatible}")
        load_audit = {
            "loading_path": "local_student_backbone_export",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "loaded_keys": sorted(loaded_state),
            "ignored_checkpoint_keys": sorted(set(source_state) - set(loaded_state)),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "export_provenance": export_audit,
        }
    else:
        model = SignalJEPA_PreLocal.from_pretrained(
            config["pretrained_repo_id"],
            revision=config["pretrained_revision"],
            strict=False,
            **kwargs,
        )
        load_audit = {
            "loading_path": "huggingface",
            "repo_id": config["pretrained_repo_id"],
            "revision": config["pretrained_revision"],
        }
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if name.startswith(("spatial_conv.", "final_layer.")):
            parameter.requires_grad = True
    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or not all(name.startswith(("spatial_conv.", "final_layer.")) for name in trainable):
        raise AssertionError(f"Unexpected trainable parameters: {trainable}")
    return model, {
        **load_audit,
        "trainable": trainable,
        "n_trainable": int(sum(trainable.values())),
    }


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def evaluate(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int) -> tuple[float, np.ndarray]:
    model.eval()
    losses, logits = [], []
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for batch_x, batch_y in _loader(x, y, batch_size, False, 0):
            output = model(batch_x.to(device))
            if isinstance(output, (tuple, list)):
                output = output[0]
            losses.append(float(criterion(output, batch_y.to(device)).cpu()))
            logits.append(output.cpu().numpy())
    values = np.concatenate(logits)
    exp = np.exp(values - values.max(axis=1, keepdims=True))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise AssertionError("Probabilities are not normalized")
    return float(sum(losses) / len(y)), probabilities


def run_fold(subject_id: int, x: np.ndarray, y: np.ndarray, split: dict, config: dict, device: torch.device) -> dict:
    fold_id = int(split["fold_id"])
    outer_train = np.asarray(split["train_indices"], dtype=np.int64)
    test = np.asarray(split["test_indices"], dtype=np.int64)
    inner_train, valid = make_inner_split(y, outer_train, config, fold_id)
    if set(test) & (set(inner_train) | set(valid)):
        raise AssertionError("Outer test leaked into fitting indices")
    normalizer = fit_normalizer(x[inner_train], config["normalization_mode"], float(config["normalization_eps"]))
    x_train = apply_normalizer(x[inner_train], normalizer)
    x_valid = apply_normalizer(x[valid], normalizer)
    x_test = apply_normalizer(x[test], normalizer)

    fold_seed = int(config["seed"]) + int(subject_id) * 100 + fold_id
    seed_everything(fold_seed)
    model, model_audit = build_model(config, x.shape[-1])
    model.to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_loss, best_epoch, best_state = float("inf"), 0, None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, int(config["n_epochs"]) + 1):
        model.train()
        train_loss, train_count = 0.0, 0
        for batch_x, batch_y in _loader(x_train, y[inner_train], int(config["batch_size"]), True, fold_seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x.to(device))
            if isinstance(output, (tuple, list)):
                output = output[0]
            loss = criterion(output, batch_y.to(device))
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * len(batch_y)
            train_count += len(batch_y)
        valid_loss, _ = evaluate(model, x_valid, y[valid], device, int(config["batch_size"]))
        history.append({"epoch": epoch, "train_loss": train_loss / train_count, "valid_loss": valid_loss})
        if valid_loss < best_loss - float(config["early_stopping_threshold"]):
            best_loss, best_epoch = valid_loss, epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(config["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was created")
    model.load_state_dict(best_state, strict=True)
    tested_state_hash = state_hash(model)
    _, probabilities = evaluate(model, x_test, y[test], device, int(config["batch_size"]))
    predictions = probabilities.argmax(axis=1)
    majority = float(np.bincount(predictions, minlength=2).max() / len(predictions))
    return {
        "subject_id": str(subject_id),
        "fold_id": fold_id,
        "train_indices": outer_train.tolist(),
        "inner_train_indices": inner_train.tolist(),
        "validation_indices": valid.tolist(),
        "test_indices": test.tolist(),
        "true_labels": y[test].tolist(),
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "accuracy": float(accuracy_score(y[test], predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test], predictions)),
        "confusion_matrix": confusion_matrix(y[test], predictions, labels=[0, 1]).tolist(),
        "prediction_histogram": np.bincount(predictions, minlength=2).tolist(),
        "collapse_diagnostics": {
            "single_class_prediction": bool(len(np.unique(predictions)) == 1),
            "majority_prediction_fraction": majority,
            "near_collapse_7_of_8": bool(majority >= float(config.get("collapse_threshold", 0.875))),
        },
        "normalization_mode": normalizer["mode"],
        "normalizer_hash": normalizer["hash"],
        "model_audit": model_audit,
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "epochs_ran": len(history),
        "tested_state_hash": tested_state_hash,
        "history": history,
        "fold_seed": fold_seed,
        "outer_test_used_for_fit": False,
        "outer_test_used_for_selection": False,
    }


def split_hash(fold_results: list[dict]) -> str:
    payload = [
        {
            "subject_id": row["subject_id"],
            "fold_id": row["fold_id"],
            "train_indices": row["train_indices"],
            "inner_train_indices": row["inner_train_indices"],
            "validation_indices": row["validation_indices"],
            "test_indices": row["test_indices"],
        }
        for row in fold_results
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def aggregate(fold_results: list[dict], labels_by_subject: dict[str, np.ndarray]) -> tuple[dict, dict, list[dict]]:
    subject_metrics, prediction_rows = {}, []
    for subject_id, labels in labels_by_subject.items():
        rows = [row for row in fold_results if row["subject_id"] == subject_id]
        seen = np.zeros(len(labels), dtype=np.int64)
        predictions = np.full(len(labels), -1, dtype=np.int64)
        probabilities = np.full((len(labels), 2), np.nan)
        for row in rows:
            indices = np.asarray(row["test_indices"], dtype=np.int64)
            seen[indices] += 1
            predictions[indices] = np.asarray(row["predictions"])
            probabilities[indices] = np.asarray(row["probabilities"])
            for position, trial_index in enumerate(indices):
                prediction_rows.append({
                    "subject_id": subject_id,
                    "trial_index": int(trial_index),
                    "fold_id": int(row["fold_id"]),
                    "y_true": int(labels[trial_index]),
                    "y_pred": int(row["predictions"][position]),
                    "probability_0": float(row["probabilities"][position][0]),
                    "probability_1": float(row["probabilities"][position][1]),
                })
        if not np.all(seen == 1) or np.any(predictions < 0) or not np.isfinite(probabilities).all():
            raise AssertionError(f"{subject_id}: invalid exact-once OOF accounting")
        subject_metrics[subject_id] = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "n_trials": int(len(labels)),
            "exact_once_oof": True,
        }
    values = np.asarray([row["balanced_accuracy"] for row in subject_metrics.values()])
    global_metrics = {
        "mean_subject_balanced_accuracy": float(values.mean()),
        "std_subject_balanced_accuracy": float(values.std()),
        "n_subjects": int(len(values)),
        "n_original_trial_predictions": int(len(prediction_rows)),
        "exact_once_oof_all_subjects": True,
        "split_hash": split_hash(fold_results),
    }
    return subject_metrics, global_metrics, prediction_rows
