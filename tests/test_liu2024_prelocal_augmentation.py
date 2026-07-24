import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "liu2024"))

import liu2024_prelocal_augmentation as augmentation  # noqa: E402


def _config() -> dict:
    return {
        "subjects_to_use": None,
        "exclude_subjects": [],
        "target_sfreq": 128,
        "mi_window_seconds": 4.0,
        "average_reference": True,
        "bandpass_hz": [0.5, 40.0],
        "normalization_mode": "none",
        "pretrained_repo_id": "braindecode/signal-jepa_without-chans",
        "pretrained_revision": "213876ea30f0764fd25c055efcb55d1d1652a371",
        "strategy": "new",
        "cv_folds": 5,
        "conditions": ["control", "gacl_rotation_noise"],
        "num_workers": 0,
        "augmentation": {
            "rotation_axis": "z",
            "rotation_max_degrees": 10.0,
            "rotation_probability": 0.5,
            "rotation_spherical_splines": True,
            "noise_std_microvolts": 2.0,
            "noise_probability": 1.0,
        },
    }


def test_locked_config_and_sensor_positions() -> None:
    config = _config()
    augmentation.validate_config(config)
    positions = augmentation.sensor_positions(config["target_sfreq"])
    assert positions.shape == (3, 29)
    assert np.isfinite(positions).all()


def test_training_augmentation_is_deterministic_and_preserves_labels() -> None:
    config = _config()
    x = np.linspace(-5, 5, 4 * 29 * 64, dtype=np.float32).reshape(4, 29, 64)
    y = np.asarray([0, 1, 0, 1], dtype=np.int64)

    transforms_a, seeds_a = augmentation.build_fold_transforms(config, 123)
    transforms_b, seeds_b = augmentation.build_fold_transforms(config, 123)
    loader_a = augmentation.make_train_loader(x, y, 4, 999, transforms_a)
    loader_b = augmentation.make_train_loader(x, y, 4, 999, transforms_b)
    batch_x_a, batch_y_a = next(iter(loader_a))
    batch_x_b, batch_y_b = next(iter(loader_b))

    assert seeds_a == seeds_b
    assert batch_x_a.shape == (4, 29, 64)
    assert torch.isfinite(batch_x_a).all()
    assert torch.equal(batch_y_a, batch_y_b)
    assert torch.equal(batch_x_a, batch_x_b)
    assert not torch.equal(batch_x_a, torch.from_numpy(x))


def test_paired_statistics_use_subjects() -> None:
    control = {
        "1": {"balanced_accuracy": 0.50, "accuracy": 0.50},
        "2": {"balanced_accuracy": 0.55, "accuracy": 0.55},
    }
    augmented = {
        "1": {"balanced_accuracy": 0.60, "accuracy": 0.60},
        "2": {"balanced_accuracy": 0.50, "accuracy": 0.50},
    }
    rows, stats = augmentation.paired_statistics(control, augmented, 100, 2026)
    assert len(rows) == 2
    assert np.isclose(stats["mean_paired_delta"], 0.025)
    assert stats["wins_ties_losses"] == {"wins": 1, "ties": 0, "losses": 1}
