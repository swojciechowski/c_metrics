from setuptools import Extension, setup
import numpy

setup(
   name='scikit-confusion_matrix',
   version='0.0.1',
   packages=['skcm'],
   ext_modules=[
    Extension(
      name="ccm",
      sources=["src/confusion_matrix.c"],
      include_dirs=[numpy.get_include()])
   ],
   install_requires=['numpy']
)