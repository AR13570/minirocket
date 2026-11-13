"""
Tests for code/minirocket_explain.py

Author : Shreyas Shinde (Student ID 26104692)

Run from the repository root:
    python -B -m unittest tests.test_explain -v

What is tested and why:
 1. COMPLETENESS: the heat map adds up exactly to the model's decision score
    (minus the intercept), for 2-class and 3-class problems and for both feature
    sets (PPV and PPV+MPV). This proves the heat map is an exact split of the
    model's own calculation, not an approximation.
 2. The recomputed convolution outputs give exactly the model's features.
 3. The feature table has one row per feature with valid kernels and dilations.
 4. Global importance and class summaries have the expected structure.
 5. The bump the classes differ by is where the heat map is strongest.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))     # code/ (same as the other tests)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))               # repository root (benchmarks/)
import minirocket_multipool as mp
from benchmarks.common import make_classifier
from minirocket_adaptive import AdaptivePoolingMiniRocket
from minirocket_explain import MiniRocketExplainer, feature_table, convolution_outputs, KERNEL_POSITIONS


def bump_dataset(n_per_class, classes=2, length=60, seed=0):
    """Class c has a bump centred at a different place (10 + 18c); plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    X, y = [], []
    for c in range(classes):
        for _ in range(n_per_class):
            X.append(1.5 * np.exp(-0.5 * ((t - (10 + 18 * c)) / 3) ** 2) + 0.2 * rng.standard_normal(length))
            y.append(c)
    return np.array(X, dtype=np.float32), np.array(y)


def model_with_feature_set(X, y, feature_set):
    """Fitted model forced to use a given feature set (to test both PPV and PPV+MPV)."""
    model = AdaptivePoolingMiniRocket(num_features=840, random_state=0).fit(X, y)
    blocks = {"PPV": ["raw_PPV"], "PPV+MPV": ["raw_PPV", "raw_MPV"]}[feature_set]
    model.chosen_feature_set_, model.chosen_blocks_ = feature_set, blocks
    F = mp.select_blocks(mp.transform(X, model.parameters_), model.parameters_, blocks)
    model.classifier_ = make_classifier(len(X), "ridge", 0).fit(F, y)
    return model


class ExplainabilityTests(unittest.TestCase):

    def test_1_heat_map_adds_up_to_the_decision_score(self):
        for classes in (2, 3):
            for feature_set in ("PPV", "PPV+MPV"):
                with self.subTest(classes=classes, feature_set=feature_set):
                    X, y = bump_dataset(8, classes=classes)
                    X_new, _ = bump_dataset(2, classes=classes, seed=9)
                    model = model_with_feature_set(X, y, feature_set)
                    explainer = MiniRocketExplainer(model, X)
                    scores = model.classifier_.decision_function(model.transform(X_new))
                    for i, x in enumerate(X_new):
                        for class_index, cls in enumerate(model.classifier_.classes_):
                            result = explainer.explain(x, target_class=cls)
                            if classes == 2:
                                expected = scores[i] if class_index == 1 else -scores[i]
                            else:
                                expected = scores[i, class_index]
                            self.assertAlmostEqual(result["decision_score"], float(expected), places=3)
                            self.assertEqual(len(result["heat"]), len(x))

    def test_2_recomputed_convolutions_match_model_features(self):
        X, y = bump_dataset(6)
        model = model_with_feature_set(X, y, "PPV")
        dilations, per_dilation, biases = model.parameters_["raw"]
        x = X[0]
        manual, f = [], 0
        for d_index, dilation in enumerate(dilations):
            C_alpha, C_gamma = convolution_outputs(x, int(dilation))
            padding = 4 * int(dilation)
            for k in range(84):
                i0, i1, i2 = KERNEL_POSITIONS[k]
                C = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2]
                if (d_index % 2 + k) % 2 != 0:
                    C = C[padding:-padding]
                for _ in range(per_dilation[d_index]):
                    manual.append(np.mean(C > biases[f]))
                    f += 1
        np.testing.assert_allclose(np.array(manual, dtype=np.float32), model.transform(X[:1])[0], atol=1e-6)

    def test_3_feature_table(self):
        X, y = bump_dataset(6)
        model = model_with_feature_set(X, y, "PPV")
        table = feature_table(model.parameters_["raw"])
        self.assertEqual(len(table["kernel"]), len(model.parameters_["raw"][2]))
        self.assertTrue(np.all((table["kernel"] >= 0) & (table["kernel"] < 84)))
        self.assertEqual(set(np.unique(table["dilation"])), set(int(d) for d in model.parameters_["raw"][0]))

    def test_4_global_importance_and_class_summaries(self):
        X, y = bump_dataset(6, classes=3)
        model = model_with_feature_set(X, y, "PPV+MPV")
        explainer = MiniRocketExplainer(model, X)
        info = explainer.global_importance()
        self.assertEqual(len(info["by_kernel"]), 84)
        self.assertEqual(set(info["by_pool"]), {"PPV", "MPV"})
        self.assertEqual(len(info["top_kernels"]), 10)
        self.assertEqual(sorted(KERNEL_POSITIONS.shape), [3, 84])
        self.assertEqual(list(info["top_kernels"][0]["weights"]).count(2), 3)
        summaries = explainer.class_summaries(X, y)
        self.assertEqual([s["class"] for s in summaries], [0, 1, 2])
        for s in summaries:
            self.assertEqual(len(s["mean_heat"]), X.shape[1])
            self.assertIn("strongest evidence", s["text"])

    def test_5_evidence_is_where_the_classes_differ(self):
        # Class 0 has a bump at t = 10, class 1 at t = 28; after t = 40 there is only noise.
        # For class-1 series, BOTH places are evidence for class 1: "bump at 28" and
        # "no bump at 10". The noise region should carry little of the evidence.
        X, y = bump_dataset(10, classes=2, seed=3)
        model = model_with_feature_set(X, y, "PPV")
        explainer = MiniRocketExplainer(model, X)
        heat = np.array([explainer.explain(x, target_class=1)["heat"] for x in X[y == 1]]).mean(axis=0)
        self.assertGreater(heat[5:16].mean(), 0)                        # near t = 10
        self.assertGreater(heat[23:34].mean(), 0)                       # near t = 28
        noise_share = np.abs(heat[40:]).sum() / np.abs(heat).sum()       # 1/3 of the series ...
        self.assertLess(noise_share, 0.2)                                # ... but < 20% of the evidence


if __name__ == "__main__":
    unittest.main()
