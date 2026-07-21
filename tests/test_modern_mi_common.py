import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "liu2024"))

from modern_mi_common import (  # noqa: E402
    augment_training_batch,
    array_hash,
    assert_source_exclusion,
    checkpoint_signature,
    load_compatible_checkpoint,
    save_checkpoint,
)


class CheckpointProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "channel_set": "liu29",
            "native_sfreq": 500,
            "target_sfreq": 128,
            "marker_channel_index": 32,
            "onset_marker_value": 2,
            "onset_plausible_range": [800, 1300],
            "window_seconds": 4.0,
            "bandpass_hz": [4.0, 40.0],
            "filter_order": 4,
            "normalization_mode": "channel_standardize",
            "normalization_eps": 1e-6,
            "model_kwargs": {},
            "source_epochs": 20,
            "source_batch_size": 64,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "seed": 2026,
        }

    def test_training_setting_changes_signature(self):
        baseline = checkpoint_signature(self.config, ["sub-01", "sub-02"], "ShallowFBCSPNet")
        changed = dict(self.config, learning_rate=1e-4)
        self.assertNotEqual(
            baseline,
            checkpoint_signature(changed, ["sub-01", "sub-02"], "ShallowFBCSPNet"),
        )

    def test_source_order_does_not_change_signature(self):
        first = checkpoint_signature(self.config, ["sub-02", "sub-01"], "ShallowFBCSPNet")
        second = checkpoint_signature(self.config, ["sub-01", "sub-02"], "ShallowFBCSPNet")
        self.assertEqual(first, second)

    def test_array_hash_tracks_normalizer_values(self):
        mean = np.zeros((1, 2, 1), dtype=np.float32)
        scale = np.ones((1, 2, 1), dtype=np.float32)
        self.assertNotEqual(array_hash(mean, scale), array_hash(mean + 1, scale))

    def test_target_cannot_appear_in_sources(self):
        with self.assertRaises(AssertionError):
            assert_source_exclusion("sub-01", ["sub-01", "sub-02"])

    def test_checkpoint_metadata_must_match_exactly(self):
        model = nn.Linear(2, 2)
        expected = {"signature": "abc", "source_subject_ids": ["sub-02"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, expected)
            state = load_compatible_checkpoint(path, expected)
            self.assertEqual(set(state), set(model.state_dict()))
            with self.assertRaises(RuntimeError):
                load_compatible_checkpoint(path, expected | {"signature": "changed"})


class TrainingAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8)

    def test_disabled_augmentation_is_exact_noop(self):
        result = augment_training_batch(self.x, {"augmentation": {"enabled": False}})
        self.assertTrue(torch.equal(result, self.x))

    def test_amplitude_scale_respects_fixed_interval(self):
        cfg = {"augmentation": {"enabled": True, "transforms": [{"name": "amplitude_scale", "probability": 1.0, "interval": [2.0, 2.0]}]}}
        result = augment_training_batch(self.x, cfg)
        self.assertTrue(torch.equal(result, self.x * 2))

    def test_relative_noise_is_finite_and_changes_signal(self):
        torch.manual_seed(2026)
        cfg = {"normalization_eps": 1e-6, "augmentation": {"enabled": True, "transforms": [{"name": "relative_gaussian_noise", "probability": 1.0, "std_fraction": 0.1}]}}
        result = augment_training_batch(self.x, cfg)
        self.assertEqual(result.shape, self.x.shape)
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(torch.equal(result, self.x))


if __name__ == "__main__":
    unittest.main()
