from setuptools import Extension, setup
import numpy

setup(
   name='cm-metrics',
   version='0.0.1',
   packages=['cm-metrics'],
   ext_modules=[
    Extension(name="cm-metrics", sources=["confusion_matrix.c"], include_dirs=[numpy.get_include()])
   ],
   install_requires=['numpy']
)