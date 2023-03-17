#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *confusion_matrix(PyObject *self, PyObject *args)
{
  size_t i;
  int toc[2][2] = { 0 };
  int equal;
  int positive;
  int offset;

  PyArrayObject *y_true = NULL;
  PyArrayObject *y_pred = NULL;

  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &y_true, &PyArray_Type, &y_pred)) {
    return NULL;
  }

  offset = y_true->strides[0];
  i = y_true->dimensions[0] * offset;

  while (i -= offset) {
    positive = y_true->data[i];
    equal = y_true->data[i] ^ y_pred->data[i];
    toc[positive][equal] += 1;
  }

  return Py_BuildValue("(iiii)", toc[0, 0], toc[0, 1], toc[1, 0], toc[1, 1]);
}

static PyMethodDef C_CM_METHODS[] = {
  {"confusion_matrix", confusion_matrix, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef C_CM_MODULE = {
  PyModuleDef_HEAD_INIT,
  "c_cm",
  NULL,
  -1,
  C_CM_METHODS
};

PyMODINIT_FUNC PyInit_c_cm(void)
{
  import_array();
  return PyModule_Create(&C_CM_MODULE);
}
