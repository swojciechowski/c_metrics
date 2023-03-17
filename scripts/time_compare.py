import numpy as np
import time

from sklearn.metrics import confusion_matrix
from strlearn.metrics import binary_confusion_matrix
from skcm import confusion_matrix as skcm

np.random.seed(1410)
N = 1000000

y_true = np.random.randint(2, size=N)
y_pred = np.random.randint(2, size=N)

start = time.time()
cm = binary_confusion_matrix(y_true, y_pred)
ex_time = time.time() - start

print('+' * 30)
print('stream-learn')
print(f"{ex_time:.6f}")
print(*cm, sep='-')

start = time.time()
cm = confusion_matrix(y_true, y_pred)
ex_time = time.time() - start

print('+' * 30)
print('scikit-learn')
print(f"{ex_time:.6f}")
print(*cm.flatten(), sep='-')

start = time.time()
cm = skcm(y_true, y_pred)
ex_time = time.time() - start

print('+' * 30)
print('scikit-cm')
print(f"{ex_time:.6f}")
print(*cm, sep='-')
