import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "liu2024"))

import liu2024_prelocal_clean as clean  # noqa: E402


LV14 = [1, 3, 7, 9, 10, 11, 14, 15, 17, 29, 31, 32, 37, 41]
TRAIN30 = [2, 4, 5, 6, 8, 13, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 33, 34, 35, 36, 38, 39, 40, 43, 44, 45, 46, 48, 49, 50]
VAL6 = [12, 19, 22, 30, 42, 47]


def _write_export(path: Path) -> None:
    torch.save(
        {
            "epoch": 3,
            "student_backbone_state_dict": {"feature_encoder.test": torch.ones(2)},
            "sfreq": 128,
            "input_window_seconds": 4.0,
            "ch_names": clean.LIU_EEG_NAMES,
            "preprocessing_config": {
                "sfreq": 128,
                "bandpass_low": 0.5,
                "bandpass_high": 40.0,
                "pretrain_duration_s": 4.0,
                "window_size_samples": 512,
                "filter_method": "fir",
                "model_input_unit": "microvolts",
            },
            "subject_split": {
                "train_subject_ids": TRAIN30,
                "val_subject_ids": VAL6,
                "excluded_subject_ids": LV14,
            },
        },
        path,
    )


def _config(path: Path) -> dict:
    return {
        "pretrained_checkpoint_path": str(path),
        "pretrained_checkpoint_sha256": clean.sha256_file(path),
        "require_best_checkpoint_filename": True,
        "require_checkpoint_sha256": True,
        "expected_ssl_train_subject_ids": TRAIN30,
        "expected_ssl_val_subject_ids": VAL6,
        "expected_ssl_excluded_subject_ids": LV14,
        "target_sfreq": 128,
        "mi_window_seconds": 4.0,
        "bandpass_hz": [0.5, 40.0],
    }


def test_validate_pretraining_export_accepts_locked_split(tmp_path: Path) -> None:
    path = tmp_path / "student_backbone_best.pt"
    _write_export(path)
    _, audit = clean.validate_pretraining_export(_config(path), 512)
    assert audit["checkpoint_epoch"] == 3
    assert audit["subject_split"]["excluded_subject_ids"] == LV14


def test_validate_pretraining_export_rejects_lv14_leakage(tmp_path: Path) -> None:
    path = tmp_path / "student_backbone_best.pt"
    _write_export(path)
    config = _config(path)
    config["expected_ssl_excluded_subject_ids"] = LV14[:-1]
    with pytest.raises(RuntimeError, match="SSL split mismatch"):
        clean.validate_pretraining_export(config, 512)
