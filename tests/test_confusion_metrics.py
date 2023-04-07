import unittest
import numpy as np
from numpy.testing import assert_array_equal

from confusion_metrics import ConfusionMetrics

class TestConfusionMetrics(unittest.TestCase):
    def test_confusion_matrix(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        expected_cmat = np.array([[1, 2], [3, 4]])

        cmat = ConfusionMetrics(y_true, y_pred).CM
        assert_array_equal(cmat, expected_cmat)

    def test_true_negative(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        expected_tn = 1

        cm = ConfusionMetrics(y_true, y_pred)
        self.assertEqual(cm.TN, expected_tn)

    def test_true_positive(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        expected_tp = 4

        cm = ConfusionMetrics(y_true, y_pred)
        self.assertEqual(cm.TP, expected_tp)

    def test_false_negative(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        expected_fn = 2

        cm = ConfusionMetrics(y_true, y_pred)
        self.assertEqual(cm.FN, expected_fn)

    def test_false_positive(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        expected_fp = 3

        cm = ConfusionMetrics(y_true, y_pred)
        self.assertEqual(cm.FP, expected_fp)

if __name__ == "__main__":
    unittest.main()
