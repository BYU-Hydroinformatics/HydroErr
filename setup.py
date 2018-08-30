# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='HydroErr',
    packages=['HydroErr'],
    version='1.2',
    description='Goodness of fit metrics for use in comparison studies, specifically for use in the field '
                'of hydrology',
    author='Wade Roberts',
    author_email='waderoberts123@gmail.com',
    url='https://github.com/waderoberts123/HydroErr',
    download_url='https://github.com/waderoberts123/Hydrostats/archive/1.2.tar.gz',
    keywords=['hydrology', 'error', 'metrics', 'comparison', 'statistics', 'forecast', 'observed'],
    classifiers=["License :: OSI Approved :: MIT License",
                 "Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: 3.5",
                 "Programming Language :: Python :: 3.6",
                 ],
    install_requires=[
        'numpy',
        'numba',
        'scipy',
    ],
    extras_require={
        'docs': [
            'sphinx',
            'sphinxcontrib-napoleon'
        ]
    },
)
