"""Execution harness for the prespecified honest Liu2024 candidate notebooks."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from liu2024_honest_candidates import (
    aggregate_oof,
    artifact_manifest,
    candidate_method_contract,
    covariance_trials,
    delay_embed_trials,
    embed_liu_channels,
    extract_braindecode_eegpt_features,
    extract_eegpt_features,
    fixed_spectral_features,
    fit_feature_probe,
    fit_riemann_fold,
    fold_result,
    load_eegpt_encoder,
    load_braindecode_eegpt,
    nonlesioned_hemisphere_indices,
    paired_statistics,
    resume_key,
    train_shallow_fold,
    temporal_motor_features,
    validate_split_manifest,
)
from liu2024_prelocal_clean import (
    LIU_EEG_NAMES,
    load_subject,
    preprocess_subject,
    seed_everything,
    sha256_file,
)
from modern_mi_common import MOTOR13


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _subject_files(root: Path, subjects: list | None) -> list[Path]:
    files = sorted(root.glob("sub-*/sub-*_task-motor-imagery_eeg.mat"))
    if subjects is not None:
        wanted = {f"sub-{int(value):02d}" for value in subjects}
        files = [path for path in files if path.parent.name in wanted]
    if not files:
        raise FileNotFoundError(f"No Liu2024 source MAT files under {root}")
    return files


def _load_data(config: dict) -> tuple[dict, dict, list[dict]]:
    data, labels, inventory = {}, {}, []
    for path in _subject_files(Path(config["source_extract_dir"]), config.get("subjects_to_use")):
        source = load_subject(path)
        subject_id = f"sub-{source['subject_id']:02d}"
        x, records = preprocess_subject(source, config)
        data[subject_id] = x
        labels[subject_id] = source["labels"]
        inventory.append({
            "subject_id": subject_id, "source_path": source["path"],
            "source_sha256": source["source_sha256"], "n_trials": len(x),
            "class_0": int((source["labels"] == 0).sum()), "class_1": int((source["labels"] == 1).sum()),
            "input_shape": list(x.shape), "marker_min": int(source["onsets"].min()),
            "marker_max": int(source["onsets"].max()), "marker_records": records,
        })
    return data, labels, inventory


def _fold_shard_path(directory: Path, family: str, method: str, subject_id: str, fold_id: int) -> Path:
    safe_method = method.replace("/", "_")
    return directory / "fold_shards" / f"{family}__{safe_method}__{subject_id}__fold-{fold_id}.json"


def _load_or_run_shard(path: Path, expected_resume_key: str, callback) -> dict:
    if path.exists():
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("resume_key") != expected_resume_key:
            raise RuntimeError(f"Refusing incompatible resume shard {path}")
        return row
    row = callback()
    row["resume_key"] = expected_resume_key
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, row)
    return row


def _run_riemann(config: dict, data: dict, labels: dict, splits: dict, directory: Path, key: str) -> list[dict]:
    motor_indices = [LIU_EEG_NAMES.index(name) for name in MOTOR13]
    rows = []
    for subject_id, x_all in data.items():
        x = x_all[:, motor_indices]
        representations = {
            "standard_covariance": covariance_trials(x, config["trace_normalize"]),
            "augmented_covariance": covariance_trials(
                delay_embed_trials(x, int(config["lag_samples"]), int(config["embedding_order"])),
                config["trace_normalize"],
            ),
        }
        for method, covariances in representations.items():
            for split in splits[subject_id]:
                train = np.asarray(split["train_indices"], dtype=int)
                test = np.asarray(split["test_indices"], dtype=int)
                path = _fold_shard_path(directory, "riemann", method, subject_id, split["fold_id"])
                def execute(method=method, covariances=covariances, split=split, train=train, test=test):
                    prediction, probability, audit = fit_riemann_fold(covariances, labels[subject_id], train, test)
                    return fold_result(method, subject_id, split["fold_id"], labels[subject_id], test, prediction, probability, audit)
                rows.append(_load_or_run_shard(path, key, execute))
    return rows


def _run_reflection(config: dict, data: dict, labels: dict, splits: dict, directory: Path, key: str) -> list[dict]:
    device = _device()
    rows = []
    for subject_id, x in data.items():
        for mode in config["arms"]:
            for split in splits[subject_id]:
                train = np.asarray(split["train_indices"], dtype=int)
                test = np.asarray(split["test_indices"], dtype=int)
                path = _fold_shard_path(directory, "reflection", mode, subject_id, split["fold_id"])
                def execute(mode=mode, split=split, train=train, test=test):
                    probabilities = []
                    audits = []
                    for seed in config["model_seeds"]:
                        _, seed_probabilities, audit = train_shallow_fold(
                            x, labels[subject_id], train, test, config, int(seed), mode, device,
                        )
                        probabilities.append(seed_probabilities)
                        audits.append(audit)
                    probability = np.mean(probabilities, axis=0)
                    prediction = probability.argmax(axis=1)
                    audit = {
                        "seed_audits": audits, "seed_aggregation": "mean_probability",
                        "fit_trial_ids": train.tolist(), "apply_trial_ids": test.tolist(),
                        "outer_test_used_for_fit": False,
                    }
                    return fold_result(mode, subject_id, split["fold_id"], labels[subject_id], test, prediction, probability, audit)
                rows.append(_load_or_run_shard(path, key, execute))
    return rows


def _run_hemiparetic_side(config: dict, data: dict, labels: dict, splits: dict, directory: Path, key: str) -> list[dict]:
    device = _device()
    side_map = {str(subject): str(side).lower() for subject, side in config["paralysis_side_by_subject"].items()}
    rows = []
    for subject_id, x in data.items():
        paralysis_side = side_map.get(subject_id)
        if paralysis_side not in {"left", "right"}:
            raise ValueError(f"{subject_id}: missing valid paralysis side")
        hemisphere_indices = nonlesioned_hemisphere_indices(paralysis_side)
        methods = {
            "full_montage_control": np.arange(len(LIU_EEG_NAMES), dtype=int),
            "nonlesioned_hemisphere": hemisphere_indices,
        }
        for method, channel_indices in methods.items():
            for split in splits[subject_id]:
                train = np.asarray(split["train_indices"], dtype=int)
                test = np.asarray(split["test_indices"], dtype=int)
                path = _fold_shard_path(directory, "hemiparetic_side", method, subject_id, split["fold_id"])
                def execute(method=method, channel_indices=channel_indices, split=split, train=train, test=test):
                    probabilities = []
                    audits = []
                    for seed in config["model_seeds"]:
                        _, seed_probabilities, audit = train_shallow_fold(
                            x, labels[subject_id], train, test, config, int(seed), "control", device,
                            channel_indices=channel_indices,
                        )
                        probabilities.append(seed_probabilities)
                        audits.append(audit)
                    probability = np.mean(probabilities, axis=0)
                    prediction = probability.argmax(axis=1)
                    audit = {
                        "seed_audits": audits,
                        "seed_aggregation": "mean_probability",
                        "paralysis_side": paralysis_side,
                        "nonlesioned_hemisphere": paralysis_side,
                        "fit_trial_ids": train.tolist(),
                        "apply_trial_ids": test.tolist(),
                        "outer_test_used_for_fit": False,
                    }
                    return fold_result(method, subject_id, split["fold_id"], labels[subject_id], test, prediction, probability, audit)
                rows.append(_load_or_run_shard(path, key, execute))
    return rows


def _run_eegpt(config: dict, data: dict, labels: dict, splits: dict, directory: Path, key: str) -> tuple[list[dict], dict]:
    device = _device()
    backend = config.get("eegpt_backend", "official_repository")
    if backend == "official_repository":
        encoder, checkpoint_audit = load_eegpt_encoder(config, device)
        channel_names = checkpoint_audit["channel_names"]
    elif backend == "braindecode_encoder_only":
        encoder, checkpoint_audit = load_braindecode_eegpt(config, device)
        channel_names = checkpoint_audit["target_channel_names"]
    else:
        raise ValueError(f"Unknown EEGPT backend {backend!r}")
    rows = []
    for subject_id, x in data.items():
        if backend == "official_repository":
            features = extract_eegpt_features(encoder, x, channel_names, device, int(config["feature_batch_size"]))
            montage_audit = {"mode": "direct_named_liu29"}
        else:
            eegpt_x, montage_audit = embed_liu_channels(x, channel_names)
            features = extract_braindecode_eegpt_features(
                encoder, eegpt_x, device, int(config["feature_batch_size"]),
            )
        spectral_features = fixed_spectral_features(x, float(config["target_sfreq"]))
        feature_path = directory / "features" / f"{subject_id}.npz"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        if not feature_path.exists():
            np.savez_compressed(feature_path, features=features, labels=labels[subject_id])
        representations = {
            "fixed_spectral_lda": spectral_features,
            "frozen_eegpt_lda": features,
        }
        for method, representation in representations.items():
            for split in splits[subject_id]:
                train = np.asarray(split["train_indices"], dtype=int)
                test = np.asarray(split["test_indices"], dtype=int)
                path = _fold_shard_path(directory, "eegpt", method, subject_id, split["fold_id"])
                def execute(method=method, representation=representation, split=split, train=train, test=test):
                    prediction, probability, audit = fit_feature_probe(representation, labels[subject_id], train, test)
                    audit["feature_hash"] = hashlib.sha256(np.ascontiguousarray(representation).tobytes()).hexdigest()
                    audit["montage_audit"] = montage_audit
                    return fold_result(method, subject_id, split["fold_id"], labels[subject_id], test, prediction, probability, audit)
                rows.append(_load_or_run_shard(path, key, execute))
    return rows, checkpoint_audit


def _run_temporal_motor(config: dict, data: dict, labels: dict, splits: dict, directory: Path, key: str) -> list[dict]:
    rows = []
    for subject_id, x in data.items():
        absolute, trajectory, feature_audit = temporal_motor_features(
            x, float(config["target_sfreq"]), LIU_EEG_NAMES,
        )
        representations = {
            "absolute_motor_power_lda": absolute,
            "temporal_trajectory_lda": trajectory,
        }
        for method, representation in representations.items():
            for split in splits[subject_id]:
                train = np.asarray(split["train_indices"], dtype=int)
                test = np.asarray(split["test_indices"], dtype=int)
                path = _fold_shard_path(directory, "temporal_motor", method, subject_id, split["fold_id"])
                def execute(method=method, representation=representation, split=split, train=train, test=test):
                    prediction, probability, audit = fit_feature_probe(representation, labels[subject_id], train, test)
                    audit["feature_hash"] = hashlib.sha256(np.ascontiguousarray(representation).tobytes()).hexdigest()
                    audit["feature_definition"] = feature_audit
                    return fold_result(method, subject_id, split["fold_id"], labels[subject_id], test, prediction, probability, audit)
                rows.append(_load_or_run_shard(path, key, execute))
    return rows


def _descriptive_comparisons(subject_metrics: pd.DataFrame, reference: str) -> pd.DataFrame:
    reference_rows = subject_metrics[subject_metrics.method == reference].set_index("subject_id")
    output = []
    for method in sorted(set(subject_metrics.method) - {reference}):
        candidate = subject_metrics[subject_metrics.method == method].set_index("subject_id").reindex(reference_rows.index)
        delta = candidate.balanced_accuracy.to_numpy() - reference_rows.balanced_accuracy.to_numpy()
        output.append({
            "method": method,
            "reference": reference,
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "losses": int((delta < 0).sum()),
            "inference_disabled": True,
        })
    return pd.DataFrame(output)


def run_candidate_experiment(config: dict, artifact_dir: Path) -> dict:
    """Run one candidate family and write a complete immutable artifact package."""
    family = config.get("candidate_family")
    closed_families = {
        "augmented_covariance_riemann",
        "reflection_equivariant_shallow",
        "frozen_eegpt_probe",
        "temporal_motor_trajectory",
    }
    if family in closed_families and not config.get("allow_closed_rerun", False):
        raise RuntimeError(
            f"CLOSED candidate family {family!r}: Full50 evaluation is complete. "
            "A rerun requires an explicit new prespecification and allow_closed_rerun=true; "
            "see workspace-root AGENTS.md section 2e."
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    implementation_path = Path(__file__).resolve()
    implementation_paths = [
        implementation_path,
        implementation_path.with_name("liu2024_honest_candidates.py"),
        implementation_path.with_name("liu2024_prelocal_clean.py"),
        implementation_path.with_name("modern_mi_common.py"),
    ]
    _atomic_json(artifact_dir / "config.json", config)
    seed_everything(int(config["seed"]))
    data, labels, inventory = _load_data(config)
    split_path = Path(config["split_manifest_path"]).resolve()
    splits, split_hash = validate_split_manifest(split_path, labels)
    key = resume_key(config, split_hash, implementation_paths)
    expected_methods, expected_reference = candidate_method_contract(family, config.get("arms"))
    checkpoint_audit = None
    if family == "augmented_covariance_riemann":
        fold_results = _run_riemann(config, data, labels, splits, artifact_dir, key)
        reference = "standard_covariance"
    elif family == "reflection_equivariant_shallow":
        fold_results = _run_reflection(config, data, labels, splits, artifact_dir, key)
        reference = "control"
    elif family == "hemiparetic_side_shallow":
        fold_results = _run_hemiparetic_side(config, data, labels, splits, artifact_dir, key)
        reference = "full_montage_control"
    elif family == "frozen_eegpt_probe":
        fold_results, checkpoint_audit = _run_eegpt(config, data, labels, splits, artifact_dir, key)
        reference = "fixed_spectral_lda"
    elif family == "temporal_motor_trajectory":
        fold_results = _run_temporal_motor(config, data, labels, splits, artifact_dir, key)
        reference = "absolute_motor_power_lda"
    else:
        raise ValueError(f"Unknown candidate_family={family!r}")
    if reference != expected_reference or sorted({row["method"] for row in fold_results}) != expected_methods:
        raise AssertionError("Executed methods do not match the declared candidate/control contract")

    predictions, subject_metrics, global_metrics = aggregate_oof(fold_results, labels)
    if subject_metrics.method.nunique() <= 1:
        comparisons = pd.DataFrame()
    elif config.get("paired_inference_enabled", True):
        comparisons = paired_statistics(subject_metrics, reference)
    else:
        comparisons = _descriptive_comparisons(subject_metrics, reference)
    pd.DataFrame([{key: value for key, value in row.items() if key != "marker_records"} for row in inventory]).to_csv(
        artifact_dir / "subject_inventory.csv", index=False,
    )
    _atomic_json(artifact_dir / "data_manifest.json", inventory)
    _atomic_json(artifact_dir / "outer_splits.json", splits)
    _atomic_json(artifact_dir / "cv_results.json", fold_results)
    predictions.to_csv(artifact_dir / "predictions.csv", index=False)
    subject_metrics.to_csv(artifact_dir / "subject_metrics.csv", index=False)
    comparisons.to_csv(artifact_dir / "paired_statistics.csv", index=False)
    _atomic_json(artifact_dir / "global_metrics.json", global_metrics)
    assertions = {
        "all_subjects_complete": len(labels) == len(config.get("subjects_to_use") or range(50)),
        "all_predictions_exactly_once": len(predictions) == len(labels) * 40 * predictions.method.nunique(),
        "outer_test_used_for_fit": False,
        "method_contract_matches": True,
        "expected_methods": expected_methods,
        "expected_reference": expected_reference,
        "split_manifest_sha256": sha256_file(split_path),
        "split_content_sha256": split_hash,
        "resume_key": key,
        "resume_identity_excludes_output_paths": key == resume_key(
            {**config, "artifact_dir": "ignored", "resume_run_dir": "ignored"}, split_hash, implementation_paths,
        ),
    }
    if not all(value for name, value in assertions.items() if name in {"all_subjects_complete", "all_predictions_exactly_once"}):
        raise AssertionError(f"Completion assertions failed: {assertions}")
    if checkpoint_audit is not None:
        _atomic_json(artifact_dir / "checkpoint_audit.json", checkpoint_audit)
    metadata = {
        "candidate_family": family, "artifact_dir": str(artifact_dir.resolve()),
        "config": config, "implementation_path": str(implementation_path),
        "implementation_sha256": {str(path): sha256_file(path) for path in implementation_paths},
        "python": sys.version, "platform": platform.platform(), "device": str(_device()),
        "split_hash": split_hash, "resume_key": key,
        "global_metrics": global_metrics, "checkpoint_audit": checkpoint_audit,
    }
    _atomic_json(artifact_dir / "run_metadata.json", metadata)
    (artifact_dir / "run.log").write_text("completed\n", encoding="utf-8")
    required_files = {
        "config.json", "run_metadata.json", "data_manifest.json", "outer_splits.json",
        "cv_results.json", "predictions.csv", "subject_metrics.csv", "global_metrics.json",
        "paired_statistics.csv", "run.log",
    }
    assertions["required_contract_files_present"] = all((artifact_dir / name).is_file() for name in required_files)
    preview_manifest = artifact_manifest(artifact_dir)
    assertions["recursive_manifest_includes_all_fold_shards"] = {
        str(path.relative_to(artifact_dir)) for path in (artifact_dir / "fold_shards").glob("*.json")
    }.issubset(preview_manifest)
    if not assertions["required_contract_files_present"] or not assertions["recursive_manifest_includes_all_fold_shards"]:
        raise AssertionError(f"Artifact contract assertions failed: {assertions}")
    _atomic_json(artifact_dir / "assertions.json", assertions)
    _atomic_json(artifact_dir / "manifest.json", artifact_manifest(artifact_dir))
    return metadata
