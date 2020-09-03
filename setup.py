#!/usr/bin/env python
import setuptools

VERSION = '1.0.0'

setuptools.setup(
    name='body_comp',
    version=VERSION,
    description='Package for training and deployment neural networks for body composition analysis of abdominal CTs',
    author='Christopher P. Bridge',
    maintainer='Christopher P. Bridge',
    url='https://gitlab.ccds.io/ml/ccds/ct_body_composition',
    platforms=['Linux'],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.16.0',
        'tensorflow<2.0',
        'matplotlib>=3.2.0',
        'scikit-image>=0.16.0',
        'scipy>=1.4.0',
        'pydicom>=1.4.1',
        'pandas>=1.0.3',
        'highdicom>=0.3.0',
    ],
    package_data={
        '': ['configs/*.json'],
    }
)
