#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *confusion_matrix(PyObject *self, PyObject *args)
{
  size_t i;
  unsigned int toc[2][2] = { 0 };
  PyArrayObject *y_true = NULL;
  PyArrayObject *y_pred = NULL;

  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &y_true, &PyArray_Type, &y_pred)) {
    return NULL;
  }

  size_t size = PyArray_SIZE(y_true);
  long int *y_true_data = PyArray_DATA(y_true);
  long int *y_pred_data = PyArray_DATA(y_pred);

  for (i = 0; i < size; i++) {
    toc[y_true_data[i]][y_pred_data[i]] += 1;
  }

  return Py_BuildValue("(OO)",
    Py_BuildValue("(ii)", toc[0][0], toc[0][1]),
    Py_BuildValue("(ii)", toc[1][0], toc[1][1])
  ); 
}

static PyMethodDef CONFUSION_MATRIX_METHODS[] = {
  {"confusion_matrix", confusion_matrix, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef CONFUSION_MATRIX_MODULE = {
  PyModuleDef_HEAD_INIT,
  "confusion_matrix",
  NULL,
  -1,
  CONFUSION_MATRIX_METHODS
};

PyMODINIT_FUNC PyInit_confusion_matrix(void)
{
  import_array();
  return PyModule_Create(&CONFUSION_MATRIX_MODULE);
}
