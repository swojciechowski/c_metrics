import unittest
import numpy as np

from skcm import confusion_matrix

class TestConfusionMatrix(unittest.TestCase):
    def test_confusion_matrix(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])
        expected_cm =((1, 1), (1, 1))

        cm = confusion_matrix(y_true, y_pred)
        self.assertEquals(cm, expected_cm)

if __name__ == '__main__':
    unittest.main()