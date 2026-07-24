"""Shared leakage-safe primitives for the Liu2024 candidate experiments.

The module deliberately contains no model selection. Candidate grids and stop/go
decisions belong in prespecified experiment configs, never in outer-test code.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyriemann.tangentspace import TangentSpace
from scipy import signal
from scipy.stats import wilcoxon
from sklearn.covariance import OAS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from safetensors.torch import load_file as load_safetensors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from liu2024_prelocal_clean import LIU_EEG_NAMES, array_hash, sha256_file
from modern_mi_common import MOTOR13, bootstrap_ci, build_model, fit_normalizer, stable_hash


EEGPT_CHANNEL_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def candidate_method_contract(family: str, reflection_arms: list[str] | None = None) -> tuple[list[str], str]:
    contracts = {
        "augmented_covariance_riemann": (["augmented_covariance", "standard_covariance"], "standard_covariance"),
        "frozen_eegpt_probe": (["fixed_spectral_lda", "frozen_eegpt_lda"], "fixed_spectral_lda"),
        "hemiparetic_side_shallow": (["full_montage_control", "nonlesioned_hemisphere"], "full_montage_control"),
        "temporal_motor_trajectory": (["absolute_motor_power_lda", "temporal_trajectory_lda"], "absolute_motor_power_lda"),
    }
    if family == "reflection_equivariant_shallow":
        arms = list(reflection_arms or [])
        if not arms or "control" not in arms:
            raise ValueError("Reflection contract requires a control arm")
        return sorted(arms), "control"
    if family not in contracts:
        raise ValueError(f"Unknown candidate family {family!r}")
    return contracts[family]


def canonical_json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_split_manifest(path: Path, labels_by_subject: dict[str, np.ndarray]) -> tuple[dict, str]:
    """Load the immutable split inventory and fail on any coverage or label drift."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not set(labels_by_subject).issubset(payload):
        raise AssertionError("Loaded subjects are missing from the split manifest")
    payload = {subject_id: payload[subject_id] for subject_id in labels_by_subject}
    for subject_id, labels in labels_by_subject.items():
        folds = payload[subject_id]
        if len(folds) != 5:
            raise AssertionError(f"{subject_id}: expected five outer folds")
        seen = np.zeros(len(labels), dtype=int)
        for fold in folds:
            train = np.asarray(fold["train_indices"], dtype=int)
            test = np.asarray(fold["test_indices"], dtype=int)
            if len(train) != 32 or len(test) != 8 or set(train) & set(test):
                raise AssertionError(f"{subject_id} fold {fold['fold_id']}: invalid 32/8 partition")
            if set(train) | set(test) != set(range(40)):
                raise AssertionError(f"{subject_id} fold {fold['fold_id']}: incomplete partition")
            if np.bincount(labels[test], minlength=2).tolist() != [4, 4]:
                raise AssertionError(f"{subject_id} fold {fold['fold_id']}: test fold is not balanced")
            seen[test] += 1
        if not np.all(seen == 1):
            raise AssertionError(f"{subject_id}: outer testing is not exactly once")
    return payload, canonical_json_hash(payload)


def reflection_permutation(channel_names: list[str] | tuple[str, ...]) -> np.ndarray:
    """Return the fixed sagittal reflection permutation for a standard 10-20 montage."""
    names = list(channel_names)
    pairs = [
        ("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8"), ("FC3", "FC4"),
        ("FT7", "FT8"), ("C3", "C4"), ("T3", "T4"), ("CP3", "CP4"),
        ("TP7", "TP8"), ("P3", "P4"), ("T5", "T6"), ("O1", "O2"),
    ]
    permutation = np.arange(len(names))
    for left, right in pairs:
        if (left in names) != (right in names):
            raise ValueError(f"Reflection montage contains only one of {left}/{right}")
        if left in names:
            li, ri = names.index(left), names.index(right)
            permutation[li], permutation[ri] = ri, li
    if not np.array_equal(permutation[permutation], np.arange(len(names))):
        raise AssertionError("Reflection must be an involution")
    return permutation


def reflect_trials(x: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    result = np.asarray(x)[:, permutation, :].copy()
    if result.shape != x.shape or not np.isfinite(result).all():
        raise AssertionError("Invalid reflected trial tensor")
    return result


def nonlesioned_hemisphere_indices(paralysis_side: str) -> np.ndarray:
    """Select the clinically non-lesioned lateral channels plus fixed midline channels."""
    side = str(paralysis_side).lower()
    if side not in {"left", "right"}:
        raise ValueError(f"Unknown paralysis side {paralysis_side!r}")
    # Hemiparesis is contralateral to the nominal lesion: left paresis -> left channels retained.
    lateral_parity = 1 if side == "left" else 0
    midline = {"Fz", "FCz", "Cz", "Pz", "Oz"}
    indices = np.asarray([
        index for index, name in enumerate(LIU_EEG_NAMES)
        if name in midline or (name[-1].isdigit() and int(name[-1]) % 2 == lateral_parity)
    ], dtype=int)
    if len(indices) != 17:
        raise AssertionError(f"Expected 17 hemisphere-plus-midline channels, got {len(indices)}")
    return indices


def _oas_covariance(trial: np.ndarray, trace_normalize: bool = True) -> np.ndarray:
    covariance = OAS(store_precision=False, assume_centered=False).fit(trial.T).covariance_
    if trace_normalize:
        covariance = covariance / np.trace(covariance)
    if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() <= 0:
        raise AssertionError("Covariance is not finite SPD")
    return covariance


def covariance_trials(x: np.ndarray, trace_normalize: bool = True) -> np.ndarray:
    return np.stack([_oas_covariance(trial, trace_normalize) for trial in x])


def delay_embed_trials(x: np.ndarray, lag_samples: int, order: int = 2) -> np.ndarray:
    """Concatenate fixed delayed copies along channels without crossing trials."""
    if order < 2 or lag_samples < 1:
        raise ValueError("Delay embedding requires order >= 2 and lag_samples >= 1")
    max_lag = lag_samples * (order - 1)
    if max_lag >= x.shape[-1]:
        raise ValueError("Delay embedding exceeds trial length")
    blocks = [x[..., max_lag - step * lag_samples:x.shape[-1] - step * lag_samples] for step in range(order)]
    result = np.concatenate(blocks, axis=1)
    if result.shape != (len(x), x.shape[1] * order, x.shape[-1] - max_lag):
        raise AssertionError("Unexpected delay-embedded shape")
    return result


def fit_riemann_fold(
    covariances: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit tangent reference, scaling, and LDA strictly on outer training rows."""
    if set(map(int, train_indices)) & set(map(int, test_indices)):
        raise AssertionError("Outer test entered Riemannian fit indices")
    tangent = TangentSpace(metric="riemann")
    train_features = tangent.fit_transform(covariances[train_indices])
    test_features = tangent.transform(covariances[test_indices])
    scaler = StandardScaler().fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    classifier.fit(train_features, labels[train_indices])
    probabilities = classifier.predict_proba(test_features)
    predictions = probabilities.argmax(axis=1)
    audit = {
        "fit_trial_ids": train_indices.astype(int).tolist(),
        "apply_trial_ids": test_indices.astype(int).tolist(),
        "outer_test_used_for_fit": False,
        "tangent_reference_hash": array_hash(tangent.reference_),
        "scaler_hash": array_hash(scaler.mean_, scaler.scale_),
    }
    return predictions, probabilities, audit


def _softmax(logits: torch.Tensor) -> np.ndarray:
    return logits.softmax(dim=1).detach().cpu().numpy()


def train_shallow_fold(
    x: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    config: dict,
    seed: int,
    mode: str,
    device: torch.device,
    channel_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train a matched Shallow arm; reflected descendants never leave training."""
    if mode not in {"control", "reflection_train", "reflection_train_tta"}:
        raise ValueError(f"Unknown reflection mode {mode}")
    selected_indices = np.arange(len(LIU_EEG_NAMES)) if channel_indices is None else np.asarray(channel_indices, dtype=int)
    if len(selected_indices) == 0 or len(set(selected_indices.tolist())) != len(selected_indices):
        raise ValueError("Selected channels must be nonempty and unique")
    if selected_indices.min() < 0 or selected_indices.max() >= len(LIU_EEG_NAMES):
        raise ValueError("Selected channel index is outside Liu29")
    if mode != "control" and not np.array_equal(selected_indices, np.arange(len(LIU_EEG_NAMES))):
        raise ValueError("Reflection modes require the complete Liu29 montage")
    selected_names = [LIU_EEG_NAMES[index] for index in selected_indices]
    selected_x = x[:, selected_indices]
    permutation = reflection_permutation(LIU_EEG_NAMES)
    mean, scale = fit_normalizer(selected_x[train_indices], "channel_standardize", config["normalization_eps"])
    normalize = lambda values: ((values - mean) / scale).astype(np.float32)
    train_x = normalize(selected_x[train_indices])
    train_y = labels[train_indices].astype(np.int64)
    if mode != "control":
        reflected_x = normalize(reflect_trials(x[train_indices], permutation))
        train_x = np.concatenate([train_x, reflected_x])
        train_y = np.concatenate([train_y, 1 - train_y])
    test_x = normalize(selected_x[test_indices])

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(
        "ShallowFBCSPNet", len(selected_names), x.shape[-1], config["target_sfreq"],
        device=device, model_kwargs=config.get("model_kwargs", {}),
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=int(config["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    started = time.perf_counter()
    model.train()
    for _ in range(int(config["n_epochs"])):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.inference_mode():
        probabilities = _softmax(model(torch.from_numpy(test_x).to(device)))
        if mode == "reflection_train_tta":
            reflected_test = normalize(reflect_trials(x[test_indices], permutation))
            reflected_probabilities = _softmax(model(torch.from_numpy(reflected_test).to(device)))[:, ::-1]
            probabilities = (probabilities + reflected_probabilities) / 2.0
    predictions = probabilities.argmax(axis=1)
    audit = {
        "fit_trial_ids": train_indices.astype(int).tolist(),
        "apply_trial_ids": test_indices.astype(int).tolist(),
        "outer_test_used_for_fit": False,
        "normalizer_hash": array_hash(mean, scale),
        "selected_channel_indices": selected_indices.tolist(),
        "selected_channel_names": selected_names,
        "reflection_permutation": permutation.tolist(),
        "reflected_descendants": 0 if mode == "control" else len(train_indices),
        "elapsed_seconds": elapsed,
        "seed": seed,
    }
    return predictions, probabilities, audit


def eegpt_channel_names() -> list[str]:
    return [EEGPT_CHANNEL_ALIASES.get(name, name).upper() for name in LIU_EEG_NAMES]


def embed_liu_channels(
    x: np.ndarray,
    target_channel_names: list[str],
) -> tuple[np.ndarray, dict]:
    """Place Liu29 channels into a fixed named target montage; absent channels remain zero."""
    source_names = eegpt_channel_names()
    normalized_targets = [name.upper() for name in target_channel_names]
    if len(set(normalized_targets)) != len(normalized_targets):
        raise ValueError("Target montage contains duplicate channel names")
    missing = sorted(set(source_names) - set(normalized_targets))
    if missing:
        raise ValueError(f"Target montage is missing Liu channels: {missing}")
    output = np.zeros((len(x), len(normalized_targets), x.shape[-1]), dtype=np.float32)
    mapping = {}
    for source_index, name in enumerate(source_names):
        target_index = normalized_targets.index(name)
        output[:, target_index] = x[:, source_index]
        mapping[name] = target_index
    if not np.isfinite(output).all():
        raise AssertionError("Non-finite EEGPT montage tensor")
    return output, {
        "source_channel_names": source_names,
        "target_channel_names": normalized_targets,
        "source_to_target_index": mapping,
        "zero_filled_target_channels": [
            name for name in normalized_targets if name not in source_names
        ],
    }


def _git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def load_eegpt_encoder(config: dict, device: torch.device) -> tuple[nn.Module, dict]:
    """Instantiate the pinned official encoder and require exact target-encoder loading."""
    repository = Path(config["repository_path"]).resolve()
    checkpoint = Path(config["checkpoint_path"]).resolve()
    expected_revision = config["repository_revision"]
    expected_digest = config["checkpoint_sha256"]
    if not repository.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("EEGPT repository and checkpoint must exist locally")
    actual_revision = _git_revision(repository)
    actual_digest = sha256_file(checkpoint)
    if actual_revision != expected_revision or actual_digest != expected_digest:
        raise RuntimeError("EEGPT repository revision or checkpoint digest mismatch")
    module_path = repository / "downstream" / "Modules" / "models" / "EEGPT_mcae_finetune.py"
    spec = importlib.util.spec_from_file_location("official_eegpt_mcae_finetune", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official EEGPT module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    channels = eegpt_channel_names()
    model = module.EEGPTClassifier(
        num_classes=0, in_channels=len(channels), img_size=[len(channels), 1024],
        patch_stride=64, use_channels_names=channels, use_mean_pooling=True,
        use_chan_conv=False, desired_time_len=1024, use_avg=False,
        use_predictor=False, use_out_proj=False,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise RuntimeError("Official EEGPT checkpoint lacks state_dict")
    prefix = "target_encoder."
    encoder_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    expected = model.target_encoder.state_dict()
    if set(encoder_state) != set(expected):
        raise RuntimeError(
            f"EEGPT target-encoder key mismatch: missing={sorted(set(expected)-set(encoder_state))}, "
            f"unexpected={sorted(set(encoder_state)-set(expected))}"
        )
    shape_errors = {key: [list(encoder_state[key].shape), list(expected[key].shape)] for key in expected if encoder_state[key].shape != expected[key].shape}
    if shape_errors:
        raise RuntimeError(f"EEGPT target-encoder shape mismatch: {shape_errors}")
    model.target_encoder.load_state_dict(encoder_state, strict=True)
    encoder = model.target_encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    audit = {
        "repository_path": str(repository), "repository_revision": actual_revision,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": actual_digest,
        "loaded_keys": sorted(encoder_state), "loaded_key_count": len(encoder_state),
        "channel_names": channels, "input_shape": [len(channels), 1024],
    }
    return encoder, audit


def load_braindecode_eegpt(config: dict, device: torch.device) -> tuple[nn.Module, dict]:
    """Strictly load the pinned Braindecode encoder-only EEGPT safetensors conversion."""
    from braindecode.models import EEGPT

    checkpoint_dir = Path(config["checkpoint_dir"]).resolve()
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Incomplete Braindecode EEGPT checkpoint at {checkpoint_dir}")
    actual_config_sha = sha256_file(config_path)
    actual_weights_sha = sha256_file(weights_path)
    if actual_config_sha != config["checkpoint_config_sha256"]:
        raise RuntimeError("EEGPT config SHA256 mismatch")
    if actual_weights_sha != config["checkpoint_sha256"]:
        raise RuntimeError("EEGPT weights SHA256 mismatch")
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    # The published tensors are encoder-only. These overrides make the instantiated
    # architecture match all saved keys exactly instead of partially loading a head.
    model_config.update({
        "chan_proj_type": "none",
        "return_encoder_output": True,
        "n_outputs": None,
    })
    model = EEGPT.from_config(model_config)
    state = load_safetensors(weights_path)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected strict EEGPT load result: {incompatible}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    target_names = [channel["ch_name"].upper() for channel in model_config["chs_info"]]
    audit = {
        "backend": "braindecode_encoder_only",
        "hub_repo_id": config["hub_repo_id"],
        "hub_revision": config["hub_revision"],
        "checkpoint_dir": str(checkpoint_dir),
        "config_sha256": actual_config_sha,
        "checkpoint_sha256": actual_weights_sha,
        "strict_loaded_keys": len(state),
        "missing_keys": [],
        "unexpected_keys": [],
        "architecture_overrides": {
            "chan_proj_type": "none",
            "return_encoder_output": True,
            "n_outputs": None,
        },
        "target_channel_names": target_names,
        "input_shape": [len(target_names), int(model_config["n_times"])],
        "sfreq": float(model_config["sfreq"]),
    }
    return model, audit


def extract_braindecode_eegpt_features(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            result = model(
                torch.from_numpy(x[start:start + batch_size]).to(device),
                return_features=True,
            )
            tokens = result["features"]
            outputs.append(tokens.mean(dim=1).cpu().numpy())
    features = np.concatenate(outputs).astype(np.float32)
    if features.ndim != 2 or len(features) != len(x) or not np.isfinite(features).all():
        raise AssertionError(f"Invalid Braindecode EEGPT feature matrix {features.shape}")
    return features


def extract_eegpt_features(
    encoder: nn.Module, x: np.ndarray, channel_names: list[str], device: torch.device, batch_size: int = 8,
) -> np.ndarray:
    """Mean-pool official target-encoder patch summaries into one vector per trial."""
    chan_ids = encoder.prepare_chan_ids(channel_names).to(device)
    outputs = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            tokens = encoder(torch.from_numpy(x[start:start + batch_size]).to(device), chan_ids)
            outputs.append(tokens.flatten(2).mean(dim=1).cpu().numpy())
    features = np.concatenate(outputs).astype(np.float32)
    if features.ndim != 2 or len(features) != len(x) or not np.isfinite(features).all():
        raise AssertionError(f"Invalid EEGPT feature matrix {features.shape}")
    return features


def fit_feature_probe(
    features: np.ndarray, labels: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    scaler = StandardScaler().fit(features[train_indices])
    classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    classifier.fit(scaler.transform(features[train_indices]), labels[train_indices])
    probabilities = classifier.predict_proba(scaler.transform(features[test_indices]))
    predictions = probabilities.argmax(axis=1)
    return predictions, probabilities, {
        "fit_trial_ids": train_indices.astype(int).tolist(),
        "apply_trial_ids": test_indices.astype(int).tolist(),
        "outer_test_used_for_fit": False,
        "scaler_hash": array_hash(scaler.mean_, scaler.scale_),
    }


def fixed_spectral_features(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Return a fixed per-channel mu/beta log-power control with no fitted view selection."""
    frequencies, power = signal.welch(
        x, fs=float(sfreq), nperseg=min(int(sfreq * 2), x.shape[-1]), axis=-1,
    )
    bands = [(8.0, 12.0), (13.0, 30.0)]
    features = []
    for low, high in bands:
        mask = (frequencies >= low) & (frequencies <= high)
        if not mask.any():
            raise AssertionError(f"No Welch bins for fixed band {low}-{high} Hz")
        features.append(np.log(np.maximum(power[..., mask].mean(axis=-1), np.finfo(float).tiny)))
    result = np.concatenate(features, axis=1).astype(np.float32)
    if result.shape != (len(x), x.shape[1] * len(bands)) or not np.isfinite(result).all():
        raise AssertionError(f"Invalid fixed spectral features {result.shape}")
    return result


def temporal_motor_features(
    x: np.ndarray,
    sfreq: float,
    channel_names: list[str] | tuple[str, ...] = LIU_EEG_NAMES,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fixed low-dimensional absolute and temporal motor-power representations."""
    names = list(channel_names)
    pairs = [("FC3", "FC4"), ("C3", "C4"), ("CP3", "CP4"), ("P3", "P4")]
    indices = [(names.index(left), names.index(right)) for left, right in pairs]
    samples_per_bin = int(round(float(sfreq)))
    if x.shape[-1] != samples_per_bin * 4:
        raise ValueError("Temporal motor features require exactly four one-second bins")
    bands = [(8.0, 12.0), (13.0, 30.0)]
    absolute_parts, trajectory_parts = [], []
    time_axis = np.arange(4, dtype=float)
    centered_time = time_axis - time_axis.mean()
    slope_denominator = float(np.square(centered_time).sum())
    for low, high in bands:
        bin_asymmetry, bin_bilateral = [], []
        for bin_index in range(4):
            window = x[..., bin_index * samples_per_bin:(bin_index + 1) * samples_per_bin]
            frequencies, power = signal.welch(
                window, fs=float(sfreq), nperseg=samples_per_bin, axis=-1,
            )
            mask = (frequencies >= low) & (frequencies <= high)
            log_power = np.log(np.maximum(power[..., mask].mean(axis=-1), np.finfo(float).tiny))
            left = np.stack([log_power[:, left_index] for left_index, _ in indices], axis=1)
            right = np.stack([log_power[:, right_index] for _, right_index in indices], axis=1)
            bin_asymmetry.append((right - left).mean(axis=1))
            bin_bilateral.append(((right + left) / 2.0).mean(axis=1))
        asymmetry = np.stack(bin_asymmetry, axis=1)
        bilateral = np.stack(bin_bilateral, axis=1)
        absolute_parts.extend([asymmetry.mean(axis=1), bilateral.mean(axis=1)])
        early_minus_late = asymmetry[:, :2].mean(axis=1) - asymmetry[:, 2:].mean(axis=1)
        slope = (asymmetry * centered_time).sum(axis=1) / slope_denominator
        temporal_sd = asymmetry.std(axis=1)
        trajectory_parts.extend([early_minus_late, slope, temporal_sd])
    absolute = np.stack(absolute_parts, axis=1).astype(np.float32)
    trajectory = np.stack(trajectory_parts, axis=1).astype(np.float32)
    if absolute.shape != (len(x), 4) or trajectory.shape != (len(x), 6):
        raise AssertionError("Unexpected temporal motor feature shape")
    if not np.isfinite(absolute).all() or not np.isfinite(trajectory).all():
        raise AssertionError("Non-finite temporal motor features")
    return absolute, trajectory, {
        "pairs": pairs,
        "bands_hz": bands,
        "bins_seconds": [[index, index + 1] for index in range(4)],
        "absolute_dimension": 4,
        "trajectory_dimension": 6,
    }


def fold_result(
    method: str, subject_id: str, fold_id: int, labels: np.ndarray, test_indices: np.ndarray,
    predictions: np.ndarray, probabilities: np.ndarray, audit: dict,
) -> dict:
    truth = labels[test_indices]
    return {
        "method": method, "subject_id": subject_id, "fold_id": int(fold_id),
        "test_indices": test_indices.astype(int).tolist(), "true_labels": truth.astype(int).tolist(),
        "predictions": predictions.astype(int).tolist(), "probabilities": probabilities.tolist(),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
        "prediction_histogram": np.bincount(predictions, minlength=2).tolist(),
        "collapse_diagnostics": {"single_class_prediction": bool(len(np.unique(predictions)) == 1)},
        "transform_audit": audit,
    }


def aggregate_oof(fold_results: list[dict], labels_by_subject: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Create exactly-once predictions and subject-level inference for every method."""
    prediction_rows = []
    for row in fold_results:
        for position, trial_index in enumerate(row["test_indices"]):
            prediction_rows.append({
                "method": row["method"], "subject_id": row["subject_id"],
                "trial_index": int(trial_index), "fold_id": int(row["fold_id"]),
                "y_true": int(row["true_labels"][position]), "y_pred": int(row["predictions"][position]),
                "probability_0": float(row["probabilities"][position][0]),
                "probability_1": float(row["probabilities"][position][1]),
            })
    predictions = pd.DataFrame(prediction_rows).sort_values(["method", "subject_id", "trial_index"])
    methods = sorted(predictions["method"].unique())
    subject_rows = []
    for method in methods:
        for subject_id, labels in labels_by_subject.items():
            rows = predictions[(predictions.method == method) & (predictions.subject_id == subject_id)]
            if rows.trial_index.tolist() != list(range(40)) or not np.array_equal(rows.y_true, labels):
                raise AssertionError(f"{method}/{subject_id}: predictions are not exactly once")
            subject_rows.append({
                "method": method, "subject_id": subject_id,
                "accuracy": float(accuracy_score(rows.y_true, rows.y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(rows.y_true, rows.y_pred)),
                "f1": float(f1_score(rows.y_true, rows.y_pred, zero_division=0)),
                "confusion_matrix": confusion_matrix(rows.y_true, rows.y_pred, labels=[0, 1]).tolist(),
            })
    subject_metrics = pd.DataFrame(subject_rows)
    global_metrics = {"methods": {}, "n_subjects": len(labels_by_subject), "n_trials": int(len(labels_by_subject) * 40)}
    for method in methods:
        rows = subject_metrics[subject_metrics.method == method]
        values = rows.balanced_accuracy.to_numpy()
        pred_rows = predictions[predictions.method == method]
        global_metrics["methods"][method] = {
            "mean_subject_balanced_accuracy": float(values.mean()),
            "subject_bootstrap_95_ci": bootstrap_ci(values, seed=202607, n_boot=10000),
            "trial_weighted_accuracy": float(accuracy_score(pred_rows.y_true, pred_rows.y_pred)),
            "trial_weighted_balanced_accuracy": float(balanced_accuracy_score(pred_rows.y_true, pred_rows.y_pred)),
            "trial_weighted_f1": float(f1_score(pred_rows.y_true, pred_rows.y_pred, zero_division=0)),
            "predicted_class_1_fraction": float(pred_rows.y_pred.mean()),
        }
    return predictions, subject_metrics, global_metrics


def paired_statistics(subject_metrics: pd.DataFrame, reference: str) -> pd.DataFrame:
    reference_rows = subject_metrics[subject_metrics.method == reference].set_index("subject_id")
    output = []
    for method in sorted(set(subject_metrics.method) - {reference}):
        candidate = subject_metrics[subject_metrics.method == method].set_index("subject_id")
        if list(candidate.index) != list(reference_rows.index):
            candidate = candidate.reindex(reference_rows.index)
        delta = candidate.balanced_accuracy.to_numpy() - reference_rows.balanced_accuracy.to_numpy()
        rng = np.random.default_rng(202607)
        bootstrap = rng.choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
        statistic, p_value = wilcoxon(delta, zero_method="zsplit", alternative="two-sided")
        output.append({
            "method": method, "reference": reference, "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "paired_bootstrap_95_ci_low": float(np.quantile(bootstrap, 0.025)),
            "paired_bootstrap_95_ci_high": float(np.quantile(bootstrap, 0.975)),
            "wilcoxon_statistic": float(statistic), "wilcoxon_p": float(p_value),
            "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()), "losses": int((delta < 0).sum()),
        })
    return pd.DataFrame(output)


def artifact_manifest(directory: Path) -> dict[str, dict]:
    manifest = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            relative = str(path.relative_to(directory))
            manifest[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return manifest


def resume_key(config: dict, split_hash: str, implementation_paths: list[Path] | tuple[Path, ...]) -> str:
    computational_config = {
        key: value for key, value in config.items() if key not in {"artifact_dir", "resume_run_dir"}
    }
    implementation_hashes = {
        str(path.resolve()): sha256_file(path.resolve()) for path in implementation_paths
    }
    return stable_hash(
        {"config": computational_config, "split_hash": split_hash, "implementation_sha256": implementation_hashes},
        length=32,
    )
