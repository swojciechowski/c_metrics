import numpy as np
import time
from tabulate import tabulate

from sklearn.metrics import balanced_accuracy_score
from confusion_metrics import ConfusionMetrics

def cm_bac(y_true, y_pred):
    return ConfusionMetrics(y_true, y_pred).balanced_accuracy

np.random.seed(1410)
N = 10000

fn_callbacks = [
    ("scikit-learn", balanced_accuracy_score),
    ("confusion-metrics", cm_bac)
]

y_true = np.random.randint(2, size=N)
y_pred = np.random.randint(2, size=N)

times = []

for fn_name, fn_cb in fn_callbacks:
    start = time.time()
    metric = fn_cb(y_true, y_pred)
    ex_time = time.time() - start
    
    print(fn_name, ':', metric)

    times.append((fn_name, ex_time))

tbl = tabulate(times, headers=["Name", "Time [s]"])
print(tbl)