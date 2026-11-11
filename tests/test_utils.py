import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import (  # noqa: E402
    evaluate_model,
    load_data,
    load_model,
    plot_confusion_matrix,
    plot_roc_curve,
    save_model,
    split_features_target,
)


class TestEvaluateModel(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([1, 1, 0, 0, 1, 0])
        self.y_pred = np.array([1, 0, 0, 0, 1, 1])

    def test_metric_values(self):
        m = evaluate_model(self.y_true, self.y_pred)
        self.assertAlmostEqual(m["Accuracy"], 4 / 6)
        self.assertAlmostEqual(m["Precision"], 2 / 3)
        self.assertAlmostEqual(m["Recall"], 2 / 3)
        self.assertAlmostEqual(m["F1"], 2 / 3)

    def test_returns_all_keys(self):
        m = evaluate_model(self.y_true, self.y_pred)
        self.assertEqual(set(m), {"Accuracy", "Precision", "Recall", "F1"})

    def test_roc_auc_included_with_scores(self):
        m = evaluate_model(self.y_true, self.y_pred, y_score=[0.9, 0.4, 0.2, 0.1, 0.8, 0.6])
        self.assertIn("ROC_AUC", m)

    def test_perfect_prediction(self):
        m = evaluate_model(self.y_true, self.y_true)
        self.assertEqual(m["Accuracy"], 1.0)


class TestPlots(unittest.TestCase):
    def test_confusion_matrix_plot_saves(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cm.png")
            plot_confusion_matrix([1, 0, 1, 0], [1, 0, 0, 0], save_path=p)
            self.assertTrue(os.path.exists(p))

    def test_roc_plot_saves(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "roc.png")
            plot_roc_curve([1, 0, 1, 0], [0.9, 0.2, 0.6, 0.4], save_path=p)
            self.assertTrue(os.path.exists(p))


class TestDataHelpers(unittest.TestCase):
    def test_load_data_semicolon_and_bool_target(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bills.csv")
            with open(p, "w") as f:
                f.write("is_genuine;diagonal;length\nTrue;171.8;112.8\nFalse;171.5;113.3\n")
            df = load_data(p)
            self.assertEqual(df["is_genuine"].tolist(), [1, 0])
            X, y = split_features_target(df)
            self.assertNotIn("is_genuine", X.columns)
            self.assertEqual(len(y), 2)

    def test_missing_target_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.csv")
            pd.DataFrame({"a": [1, 2]}).to_csv(p, index=False)
            with self.assertRaises(KeyError):
                load_data(p)


class TestPickleModel(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pkl")
            save_model({"a": 1}, p)
            self.assertEqual(load_model(p), {"a": 1})


if __name__ == "__main__":
    unittest.main()
