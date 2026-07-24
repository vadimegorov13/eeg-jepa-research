import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "liu2024"))

from liu2024_honest_candidates import (  # noqa: E402
    artifact_manifest,
    candidate_method_contract,
    delay_embed_trials,
    eegpt_channel_names,
    embed_liu_channels,
    fixed_spectral_features,
    nonlesioned_hemisphere_indices,
    reflect_trials,
    reflection_permutation,
    resume_key,
    temporal_motor_features,
    validate_split_manifest,
)
from liu2024_prelocal_clean import LIU_EEG_NAMES  # noqa: E402
from liu2024_candidate_runner import run_candidate_experiment  # noqa: E402


class ClosedExperimentTests(unittest.TestCase):
    def test_completed_candidate_families_fail_before_artifact_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "closed-run"
            with self.assertRaisesRegex(RuntimeError, "CLOSED candidate family"):
                run_candidate_experiment(
                    {"candidate_family": "temporal_motor_trajectory"},
                    artifact_dir,
                )
            self.assertFalse(artifact_dir.exists())


class ReflectionTests(unittest.TestCase):
    def test_reflection_is_involution_and_fixes_midline(self):
        permutation = reflection_permutation(LIU_EEG_NAMES)
        self.assertTrue(np.array_equal(permutation[permutation], np.arange(29)))
        for channel in ["Fz", "FCz", "Cz", "Pz", "Oz"]:
            index = LIU_EEG_NAMES.index(channel)
            self.assertEqual(permutation[index], index)

    def test_reflection_round_trip(self):
        x = np.arange(2 * 29 * 8).reshape(2, 29, 8)
        permutation = reflection_permutation(LIU_EEG_NAMES)
        self.assertTrue(np.array_equal(reflect_trials(reflect_trials(x, permutation), permutation), x))

    def test_eegpt_aliases_are_unique(self):
        names = eegpt_channel_names()
        self.assertEqual(len(names), 29)
        self.assertEqual(len(set(names)), 29)
        self.assertIn("T7", names)
        self.assertNotIn("T3", names)

    def test_nonlesioned_hemisphere_selection_is_fixed_and_symmetric(self):
        left = nonlesioned_hemisphere_indices("left")
        right = nonlesioned_hemisphere_indices("right")
        self.assertEqual(len(left), 17)
        self.assertEqual(len(right), 17)
        left_names = {LIU_EEG_NAMES[index] for index in left}
        right_names = {LIU_EEG_NAMES[index] for index in right}
        self.assertIn("C3", left_names)
        self.assertNotIn("C4", left_names)
        self.assertIn("C4", right_names)
        self.assertNotIn("C3", right_names)
        self.assertEqual(left_names & right_names, {"Fz", "FCz", "Cz", "Pz", "Oz"})

    def test_liu_to_eegpt_mapping_preserves_named_channels(self):
        source = np.arange(2 * 29 * 4, dtype=np.float32).reshape(2, 29, 4)
        targets = eegpt_channel_names() + ["AF3", "POZ"]
        mapped, audit = embed_liu_channels(source, targets)
        self.assertEqual(mapped.shape, (2, 31, 4))
        self.assertTrue(np.array_equal(mapped[:, :29], source))
        self.assertTrue(np.all(mapped[:, 29:] == 0))
        self.assertEqual(audit["zero_filled_target_channels"], ["AF3", "POZ"])


class DelayEmbeddingTests(unittest.TestCase):
    def test_embedding_never_crosses_trials(self):
        x = np.stack([np.full((2, 10), value) for value in [1.0, 2.0]])
        embedded = delay_embed_trials(x, lag_samples=2, order=2)
        self.assertEqual(embedded.shape, (2, 4, 8))
        self.assertTrue(np.all(embedded[0] == 1.0))
        self.assertTrue(np.all(embedded[1] == 2.0))

    def test_fixed_spectral_control_is_finite_and_fixed_size(self):
        time = np.arange(512) / 128.0
        x = np.stack([np.stack([np.sin(2 * np.pi * (10 + channel % 2) * time) for channel in range(29)])] * 3)
        features = fixed_spectral_features(x.astype(np.float32), 128)
        self.assertEqual(features.shape, (3, 58))
        self.assertTrue(np.isfinite(features).all())

    def test_temporal_motor_features_have_prespecified_dimensions(self):
        rng = np.random.default_rng(2026)
        x = rng.normal(size=(5, 29, 512)).astype(np.float32)
        absolute, trajectory, audit = temporal_motor_features(x, 128)
        self.assertEqual(absolute.shape, (5, 4))
        self.assertEqual(trajectory.shape, (5, 6))
        self.assertEqual(audit["pairs"][1], ("C3", "C4"))


class SplitTests(unittest.TestCase):
    def test_manifest_requires_exact_once_balanced_folds(self):
        labels = np.tile([0, 1], 20)
        folds = []
        for fold_id in range(5):
            test = np.array([2 * fold_id, 2 * fold_id + 1, 10 + 2 * fold_id, 11 + 2 * fold_id,
                             20 + 2 * fold_id, 21 + 2 * fold_id, 30 + 2 * fold_id, 31 + 2 * fold_id])
            train = np.setdiff1d(np.arange(40), test)
            folds.append({"fold_id": fold_id, "train_indices": train.tolist(), "test_indices": test.tolist()})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            path.write_text(__import__("json").dumps({"sub-01": folds}), encoding="utf-8")
            payload, digest = validate_split_manifest(path, {"sub-01": labels})
            self.assertEqual(len(payload["sub-01"]), 5)
            self.assertEqual(len(digest), 64)


class ProvenanceTests(unittest.TestCase):
    def test_resume_identity_covers_implementations_but_not_output_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(4):
                path = Path(directory) / f"implementation_{index}.py"
                path.write_text(f"value = {index}\n", encoding="utf-8")
                paths.append(path)
            config = {"candidate_family": "augmented_covariance_riemann", "artifact_dir": "a", "resume_run_dir": None}
            baseline = resume_key(config, "split", paths)
            relocated = resume_key({**config, "artifact_dir": "b", "resume_run_dir": "old"}, "split", paths)
            self.assertEqual(baseline, relocated)
            paths[2].write_text("value = 'changed'\n", encoding="utf-8")
            self.assertNotEqual(baseline, resume_key(config, "split", paths))

    def test_recursive_manifest_includes_nested_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "fold_shards"
            nested.mkdir()
            (root / "config.json").write_text("{}", encoding="utf-8")
            (nested / "fold.json").write_text("{}", encoding="utf-8")
            manifest = artifact_manifest(root)
            self.assertIn("config.json", manifest)
            self.assertIn("fold_shards/fold.json", manifest)

    def test_eegpt_contract_requires_same_fold_spectral_control(self):
        methods, reference = candidate_method_contract("frozen_eegpt_probe")
        self.assertEqual(methods, ["fixed_spectral_lda", "frozen_eegpt_lda"])
        self.assertEqual(reference, "fixed_spectral_lda")

    def test_hemiparetic_side_contract_requires_full_montage_control(self):
        methods, reference = candidate_method_contract("hemiparetic_side_shallow")
        self.assertEqual(methods, ["full_montage_control", "nonlesioned_hemisphere"])
        self.assertEqual(reference, "full_montage_control")


if __name__ == "__main__":
    unittest.main()
