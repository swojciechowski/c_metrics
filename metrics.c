#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *metrics(PyObject *self, PyObject *args)
{
  PyArrayObject *y_true = NULL;
  PyArrayObject *y_pred = NULL;
  int tp = 0, tn = 0, fp = 0, fn = 0;
  size_t i;
  int offset = 1;

  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &y_true, &PyArray_Type, &y_pred)) {
    return NULL;
  }

  offset = y_true->strides[0];
  i = y_true->dimensions[0] * offset;

  while (i -= offset) {
    // printf("%x | %x | %x \r\n", y_true->data[i], y_pred->data[i], y_true->data[i] & y_pred->data[i]);
    tp += y_true->data[i] & y_pred->data[i];
    tn += !y_true->data[i] & !y_pred->data[i];
    fp += !y_true->data[i] & y_pred->data[i];
    fn += y_true->data[i] & !y_pred->data[i];
  }

  return Py_BuildValue("(iiii)", tn, fp, fn, tp);
}

static PyMethodDef METRICS_METHODS[] = {
  {"metrics", metrics, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef METRICS_MODULE = {
  PyModuleDef_HEAD_INIT,
  "metrics",
  NULL,
  -1,
  METRICS_METHODS
};

PyMODINIT_FUNC PyInit_metrics(void)
{
  import_array();
  return PyModule_Create(&METRICS_MODULE);
}
