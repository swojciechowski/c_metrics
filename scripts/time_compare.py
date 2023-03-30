import numpy as np
import time
from tabulate import tabulate

from sklearn.metrics import confusion_matrix as sklrn_cm
from strlearn.metrics import binary_confusion_matrix as stlrn_cm
from skcm import confusion_matrix as skcm_cm

np.random.seed(1410)
N = 1000000

cms = [
    ("scikit-learn", sklrn_cm),
    ("stream-learn", stlrn_cm),
    ("scikit-metrics", skcm_cm)
]

y_true = np.random.randint(2, size=N)
y_pred = np.random.randint(2, size=N)

times = []

for cm_name, cm_cb in cms:
    start = time.time()
    cm = cm_cb(y_true, y_pred)
    ex_time = time.time() - start

    times.append((cm_name, ex_time))

tbl = tabulate(times, headers=["Name", "Time [s]"])
print(tbl)