from distutils.core import setup, Extension

m_metrics = Extension('metrics', sources = ['metrics.c'])

setup(
    name = 'metrics',
    version = '1.0',
    description = 'This is a demo package',
    ext_modules = [m_metrics]
)
