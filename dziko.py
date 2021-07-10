import numpy as np
import time

from sklearn.metrics import confusion_matrix
from strlearn.metrics import binary_confusion_matrix
from metrics import metrics

np.random.seed(1410)
N = 1000000

y_true = np.random.randint(2, size=N)
y_pred = np.random.randint(2, size=N)

start = time.time()
cm = binary_confusion_matrix(y_true, y_pred)
ex_time = time.time() - start

print('+' * 30)
print('strlearn')
print(f"{ex_time:.6f}")
print(*cm, sep='-')

start = time.time()
cm = confusion_matrix(y_true, y_pred)
ex_time = time.time() - start

print('+' * 30)
print('sklearn')
print(f"{ex_time:.6f}")
print(*cm.flatten(), sep='-')

start = time.time()
cm = metrics(y_true, y_pred)
ex_time = time.time() - start

print('+' * 30)
print('metrics')
print(f"{ex_time:.6f}")
print(*cm, sep='-')
