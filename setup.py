from setuptools import Extension, setup
import numpy

setup(
   name='confusion-metrics',
   version='0.0.1',
   packages=['confusion_metrics'],
   ext_modules=[
    Extension(
      name="confusion_matrix",
      sources=["src/confusion_matrix/confusion_matrix.c"],
      include_dirs=[numpy.get_include()],
      define_macros=[])
   ],
   install_requires=['numpy']
)