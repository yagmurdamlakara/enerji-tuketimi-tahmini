import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from clean_data_rerun import CONFIG, MODELS, SEEDS
from common_test_comparison import evaluate
from reference_baselines import TARGET


class CommonComparisonChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        index = pd.date_range("2020-01-01", periods=250, freq="h")
        self.hourly = pd.DataFrame(
            {TARGET: np.sin(np.arange(250) / 5) + 2}, index=index
        ).drop(index[40])
        self.targets = pd.DataFrame(
            {
                "target_timestamp": index[200:240],
                "actual_kwh": self.hourly.loc[index[200:240], TARGET].to_numpy(),
            }
        )
        path = self.root / "model_test_targets.csv"
        self.targets.to_csv(path, index=False)
        summary = dict(
            CONFIG,
            status="complete",
            completed_runs=20,
            minimum_valid_minutes_per_variable=60,
            raw_sha256="fixture",
            test_targets_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            split_counts=[dict(split="test", samples=40)],
        )
        (self.root / "run_summary.json").write_text(json.dumps(summary))
        for model in MODELS:
            for seed in SEEDS:
                frame = self.targets.assign(
                    prediction_kwh=self.targets.actual_kwh + seed / 100,
                    model=model,
                    seed=seed,
                )
                frame.to_csv(
                    self.root / f"{model}_seed_{seed}_predictions.csv", index=False
                )

    def test_exact_lag_intersection_and_metric_mean(self):
        table, seeds, variability, shared, audit = evaluate(self.hourly, self.root)
        self.assertEqual(audit["common_test_targets"], 39)
        self.assertEqual(audit["missing_history_by_method"]["weekly_168h"], 1)
        self.assertNotIn(pd.Timestamp("2020-01-09 16:00:00"), shared.index)
        self.assertEqual(table.evaluated_targets.tolist(), [39] * 7)
        self.assertEqual(len(seeds), 20)
        self.assertEqual(len(variability), 4)
        mlp = table.set_index("method").loc["MLP"]
        self.assertAlmostEqual(mlp.RMSE_kWh, 0.33)
        y = shared.actual_kwh.to_numpy()
        denominator = np.sum((y - y.mean()) ** 2)
        expected_r2 = 1 - len(y) * np.mean((np.array(SEEDS) / 100) ** 2) / denominator
        self.assertAlmostEqual(mlp.R2, expected_r2)
        ensemble_r2 = 1 - len(y) * 0.33**2 / denominator
        self.assertNotAlmostEqual(mlp.R2, ensemble_r2)
        weekly = (
            self.hourly[TARGET]
            .reindex(shared.index - pd.Timedelta(hours=168))
            .to_numpy()
        )
        np.testing.assert_allclose(shared.weekly_168h_prediction_kwh, weekly)

    def test_missing_seed_is_rejected(self):
        (self.root / "MLP_seed_11_predictions.csv").unlink()
        with self.assertRaisesRegex(ValueError, "missing="):
            evaluate(self.hourly, self.root)

    def test_changed_actual_is_rejected(self):
        path = self.root / "GRU_seed_22_predictions.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "actual_kwh"] += 0.01
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Actual energy differs"):
            evaluate(self.hourly, self.root)

    def test_missing_model_timestamp_is_not_silently_intersected(self):
        path = self.root / "LSTM_seed_55_predictions.csv"
        pd.read_csv(path).iloc[1:].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Target timestamps differ"):
            evaluate(self.hourly, self.root)

    def test_duplicate_timestamp_is_rejected(self):
        path = self.root / "BiLSTM_seed_44_predictions.csv"
        frame = pd.read_csv(path)
        frame.loc[1, "target_timestamp"] = frame.loc[0, "target_timestamp"]
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate(self.hourly, self.root)


if __name__ == "__main__":
    unittest.main()
