from setuptools import Extension, setup
import numpy

setup(
   name='scikit-confusion_matrix',
   version='0.0.1',
   packages=['skcm'],
   ext_modules=[
    Extension(
      name="confusion_matrix",
      sources=["confusion_matrix/confusion_matrix.c"],
      include_dirs=[numpy.get_include()],
      define_macros=[])
   ],
   install_requires=['numpy']
)