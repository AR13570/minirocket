"""
test_service.py - tests for the UI/API service layer (no Streamlit or FastAPI needed)

Author : Shreyas Shinde (Student ID 26104692)

Run from the repository root:
    python -B -m unittest discover -s ui -p "test_*.py" -v

What is tested and why:
 1. A bundle can be built, saved to disk and loaded back with identical predictions
    (this is what the UI/API do on start-up).
 2. predict / explain / early_timeline return complete, consistent answers
    (the explanation's heat map adds up to the model's decision score).
 3. Input handling: series of a different length are resampled, bad input is
    rejected, CSV files with a header row or a column layout are read correctly.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import service


def small_dataset(seed=0, length=40):
    """Two classes with a bump at different places, small enough for fast tests."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    X, y = [], []
    for label, centre in (("A", 10), ("B", 28)):
        for _ in range(10):
            X.append(np.exp(-0.5 * ((t - centre) / 3) ** 2) + 0.2 * rng.standard_normal(length))
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)


class ServiceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        X_train, y_train = small_dataset(0)
        X_test, y_test = small_dataset(1)
        cls.bundle = service.build_bundle_from_arrays("Toy", X_train, y_train, X_test, y_test,
                                                      num_features=840, max_background=10,
                                                      prefix_fractions=(0.3, 0.6, 1.0))

    def test_1_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = service.save_bundle(self.bundle, folder)
            self.assertTrue(path.exists() and path.with_suffix(".json").exists())
            loaded = service.load_bundle("Toy", folder=folder, build_if_missing=False)
            np.testing.assert_array_equal(loaded.model.predict(self.bundle.X_test),
                                          self.bundle.model.predict(self.bundle.X_test))
            with self.assertRaises(FileNotFoundError):
                service.load_bundle("Missing", folder=folder, build_if_missing=False)

    def test_2_metadata_and_summaries(self):
        m = self.bundle.metadata
        for key in ("test_accuracy", "chosen_feature_set", "early_metrics", "classes", "created"):
            self.assertIn(key, m)
        self.assertEqual(m["classes"], ["A", "B"])
        self.assertEqual(len(self.bundle.class_summaries), 2)

    def test_3_predict_explain_early(self):
        x = self.bundle.X_test[0]
        p = service.predict(self.bundle, x)
        self.assertIn(p["predicted_class"], ["A", "B"])
        self.assertEqual(set(p["scores"]), {"A", "B"})
        self.assertEqual(max(p["scores"], key=p["scores"].get), p["predicted_class"])

        e = service.explain(self.bundle, x)
        self.assertEqual(len(e["heat"]), len(x))
        self.assertEqual(e["predicted_class"], p["predicted_class"])
        # heat adds up to the decision score (minus intercept) for the explained class
        e_b = service.explain(self.bundle, x, target_class="B")
        self.assertAlmostEqual(sum(e_b["heat"]) + self.bundle.explainer.intercept[0], p["scores"]["B"], places=3)

        timeline = service.early_timeline(self.bundle, x)
        self.assertEqual([s["points_seen"] for s in timeline["steps"]], [12, 24, 40])
        self.assertTrue(timeline["decision"]["confident"])
        # full length always answers: threshold shown as 0.0 (not None = "never stops here")
        self.assertEqual(timeline["steps"][-1]["threshold"], 0.0)
        self.assertTrue(timeline["steps"][-1]["confident"])

    def test_4_input_handling(self):
        x, note = service.prepare_series(np.linspace(0, 1, 20), 40)
        self.assertEqual(len(x), 40)
        self.assertIn("resampled", note)
        self.assertEqual(service.prepare_series(np.ones(40), 40)[1], "")
        with self.assertRaises(ValueError):
            service.prepare_series([1.0, float("nan"), 2.0], 40)
        np.testing.assert_array_equal(service.parse_series_csv("value\n1\n2\n3\n"), [1, 2, 3])
        np.testing.assert_array_equal(service.parse_series_csv(b"1.5,2.5,3.5"), [1.5, 2.5, 3.5])
        with self.assertRaises(ValueError):
            service.parse_series_csv("no numbers here")


if __name__ == "__main__":
    unittest.main()
