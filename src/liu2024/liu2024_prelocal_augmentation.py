"""Matched training-only augmentation for the clean Liu2024 PreLocal pipeline."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import torch
from braindecode.augmentation import AugmentedDataLoader, GaussianNoise, SensorsRotation
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

import liu2024_prelocal_clean as clean


CONTROL = "control"
AUGMENTED = "gacl_rotation_noise"
CONDITIONS = (CONTROL, AUGMENTED)


def component_hash(model: torch.nn.Module, prefixes: Sequence[str]) -> str:
    digest = hashlib.sha256()
    matched = 0
    for name, tensor in sorted(model.state_dict().items()):
        if not name.startswith(tuple(prefixes)):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
        matched += 1
    if not matched:
        raise AssertionError(f"No model state matched prefixes={tuple(prefixes)}")
    return digest.hexdigest()


def validate_config(config: dict, *, require_full50: bool = True) -> None:
    if require_full50 and (config.get("subjects_to_use") is not None or config.get("exclude_subjects")):
        raise RuntimeError("The locked comparison requires all 50 subjects")
    expected = {
        "target_sfreq": 128,
        "mi_window_seconds": 4.0,
        "average_reference": True,
        "normalization_mode": "none",
        "pretrained_repo_id": "braindecode/signal-jepa_without-chans",
        "pretrained_revision": "213876ea30f0764fd25c055efcb55d1d1652a371",
        "strategy": "new",
        "cv_folds": 5,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if list(config.get("bandpass_hz", [])) != [0.5, 40.0]:
        mismatches["bandpass_hz"] = (config.get("bandpass_hz"), [0.5, 40.0])
    if tuple(config.get("conditions", ())) != CONDITIONS:
        mismatches["conditions"] = (config.get("conditions"), list(CONDITIONS))
    augmentation = config.get("augmentation", {})
    expected_augmentation = {
        "rotation_axis": "z",
        "rotation_max_degrees": 10.0,
        "rotation_probability": 0.5,
        "rotation_spherical_splines": True,
        "noise_std_microvolts": 2.0,
        "noise_probability": 1.0,
    }
    for key, value in expected_augmentation.items():
        if augmentation.get(key) != value:
            mismatches[f"augmentation.{key}"] = (augmentation.get(key), value)
    if int(config.get("num_workers", -1)) != 0:
        mismatches["num_workers"] = (config.get("num_workers"), 0)
    if mismatches:
        raise RuntimeError(f"Locked clean augmentation config mismatch: {mismatches}")


def sensor_positions(sfreq: float) -> np.ndarray:
    positions = np.stack([channel["loc"][:3] for channel in clean.make_info(sfreq)["chs"]], axis=1)
    expected_shape = (3, len(clean.LIU_EEG_NAMES))
    if positions.shape != expected_shape or not np.isfinite(positions).all():
        raise ValueError(f"Invalid Liu29 sensor positions: {positions.shape}")
    return positions


def build_fold_transforms(config: dict, fold_seed: int) -> tuple[list, dict]:
    spec = config["augmentation"]
    rotation_seed = int(fold_seed) + 10_000
    noise_seed = int(fold_seed) + 20_000
    transforms = [
        SensorsRotation(
            probability=float(spec["rotation_probability"]),
            sensors_positions_matrix=sensor_positions(config["target_sfreq"]),
            axis=str(spec["rotation_axis"]),
            max_degrees=float(spec["rotation_max_degrees"]),
            spherical_splines=bool(spec["rotation_spherical_splines"]),
            random_state=rotation_seed,
        ),
        GaussianNoise(
            probability=float(spec["noise_probability"]),
            std=float(spec["noise_std_microvolts"]),
            random_state=noise_seed,
        ),
    ]
    return transforms, {"rotation_seed": rotation_seed, "noise_seed": noise_seed}


def make_train_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle_seed: int,
    transforms: list | None,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)))
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": True,
        "generator": torch.Generator().manual_seed(int(shuffle_seed)),
        "num_workers": int(num_workers),
    }
    if transforms:
        return AugmentedDataLoader(dataset, transforms=transforms, **kwargs)
    return DataLoader(dataset, **kwargs)


def run_fold(
    condition: str,
    subject_id: int,
    x: np.ndarray,
    y: np.ndarray,
    split: dict,
    config: dict,
    device: torch.device,
) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    fold_id = int(split["fold_id"])
    outer_train = np.asarray(split["train_indices"], dtype=np.int64)
    test = np.asarray(split["test_indices"], dtype=np.int64)
    inner_train, valid = clean.make_inner_split(y, outer_train, config, fold_id)
    if set(test) & (set(inner_train) | set(valid)):
        raise AssertionError("Outer test leaked into fitting indices")

    normalizer = clean.fit_normalizer(
        x[inner_train], config["normalization_mode"], float(config["normalization_eps"])
    )
    x_train = clean.apply_normalizer(x[inner_train], normalizer)
    x_valid = clean.apply_normalizer(x[valid], normalizer)
    x_test = clean.apply_normalizer(x[test], normalizer)
    valid_hash_before = clean.array_hash(x_valid, y[valid])
    test_hash_before = clean.array_hash(x_test, y[test])

    fold_seed = int(config["seed"]) + int(subject_id) * 100 + fold_id
    clean.seed_everything(fold_seed)
    model, model_audit = clean.build_model(config, x.shape[-1])
    model.to(device)
    initial_state_hash = clean.state_hash(model)
    initial_frozen_hash = component_hash(model, ("feature_encoder.",))
    initial_trainable_hash = component_hash(model, ("spatial_conv.", "final_layer."))

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = torch.nn.CrossEntropyLoss()
    transforms, transform_seeds = (None, {})
    if condition == AUGMENTED:
        transforms, transform_seeds = build_fold_transforms(config, fold_seed)

    best_loss, best_epoch, best_state = float("inf"), 0, None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, int(config["n_epochs"]) + 1):
        model.train()
        train_loss, train_count = 0.0, 0
        train_loader = make_train_loader(
            x_train,
            y[inner_train],
            config["batch_size"],
            fold_seed + epoch,
            transforms,
            config["num_workers"],
        )
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x.to(device))
            if isinstance(output, (tuple, list)):
                output = output[0]
            loss = criterion(output, batch_y.to(device))
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * len(batch_y)
            train_count += len(batch_y)
        valid_loss, _ = clean.evaluate(model, x_valid, y[valid], device, int(config["batch_size"]))
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
    tested_state_hash = clean.state_hash(model)
    final_frozen_hash = component_hash(model, ("feature_encoder.",))
    if final_frozen_hash != initial_frozen_hash:
        raise AssertionError("Frozen feature encoder changed during downstream training")
    if clean.array_hash(x_valid, y[valid]) != valid_hash_before:
        raise AssertionError("Validation data changed during training")
    if clean.array_hash(x_test, y[test]) != test_hash_before:
        raise AssertionError("Test data changed during training")

    _, probabilities = clean.evaluate(model, x_test, y[test], device, int(config["batch_size"]))
    predictions = probabilities.argmax(axis=1)
    majority = float(np.bincount(predictions, minlength=2).max() / len(predictions))
    augmentation_audit = {
        "enabled": condition == AUGMENTED,
        "specification": config["augmentation"] if condition == AUGMENTED else None,
        **transform_seeds,
        "training_loader_class": "AugmentedDataLoader" if condition == AUGMENTED else "DataLoader",
        "validation_loader_class": "DataLoader",
        "test_loader_class": "DataLoader",
        "validation_augmented": False,
        "test_augmented": False,
        "fixed_materialized_descendants": 0,
    }
    return {
        "condition": condition,
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
        "augmentation_audit": augmentation_audit,
        "initial_state_hash": initial_state_hash,
        "initial_frozen_hash": initial_frozen_hash,
        "initial_trainable_hash": initial_trainable_hash,
        "final_frozen_hash": final_frozen_hash,
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "epochs_ran": len(history),
        "tested_state_hash": tested_state_hash,
        "history": history,
        "fold_seed": fold_seed,
        "outer_test_used_for_fit": False,
        "outer_test_used_for_selection": False,
    }


def assert_matched_pair(control: dict, augmented: dict) -> None:
    for key in (
        "subject_id",
        "fold_id",
        "train_indices",
        "inner_train_indices",
        "validation_indices",
        "test_indices",
        "true_labels",
        "fold_seed",
        "initial_state_hash",
        "initial_frozen_hash",
        "initial_trainable_hash",
    ):
        if control[key] != augmented[key]:
            raise AssertionError(f"Matched arms differ for {key}")
    if control["augmentation_audit"]["enabled"] or not augmented["augmentation_audit"]["enabled"]:
        raise AssertionError("Matched pair has invalid augmentation assignment")


def bootstrap_ci(values: np.ndarray, iterations: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(iterations), len(values)))
    means = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def paired_statistics(
    control_subjects: dict,
    augmented_subjects: dict,
    iterations: int,
    seed: int,
) -> tuple[list[dict], dict]:
    subject_ids = sorted(control_subjects, key=int)
    if subject_ids != sorted(augmented_subjects, key=int):
        raise AssertionError("Condition subject sets differ")
    rows = []
    for subject_id in subject_ids:
        control_ba = float(control_subjects[subject_id]["balanced_accuracy"])
        augmented_ba = float(augmented_subjects[subject_id]["balanced_accuracy"])
        rows.append({
            "subject_id": subject_id,
            "control_balanced_accuracy": control_ba,
            "augmented_balanced_accuracy": augmented_ba,
            "delta_augmented_minus_control": augmented_ba - control_ba,
            "control_accuracy": float(control_subjects[subject_id]["accuracy"]),
            "augmented_accuracy": float(augmented_subjects[subject_id]["accuracy"]),
        })
    control = np.asarray([row["control_balanced_accuracy"] for row in rows])
    augmented = np.asarray([row["augmented_balanced_accuracy"] for row in rows])
    delta = augmented - control
    try:
        wilcoxon_p = float(wilcoxon(delta).pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    stats = {
        "control_mean_subject_balanced_accuracy": float(control.mean()),
        "control_subject_bootstrap_95_ci": bootstrap_ci(control, iterations, seed),
        "augmented_mean_subject_balanced_accuracy": float(augmented.mean()),
        "augmented_subject_bootstrap_95_ci": bootstrap_ci(augmented, iterations, seed + 1),
        "mean_paired_delta": float(delta.mean()),
        "median_paired_delta": float(np.median(delta)),
        "paired_subject_bootstrap_95_ci": bootstrap_ci(delta, iterations, seed + 2),
        "wilcoxon_p": wilcoxon_p,
        "wins_ties_losses": {
            "wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "losses": int((delta < 0).sum()),
        },
    }
    return rows, stats
