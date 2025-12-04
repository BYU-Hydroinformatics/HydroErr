# -*- coding: utf-8 -*-

from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="HydroErr",
    packages=["HydroErr"],
    version="1.24",
    description="Goodness of fit metrics for use in comparison studies, specifically for use in the field "
    "of hydrology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Wade Roberts",
    author_email="waderoberts123@gmail.com",
    url="https://github.com/BYU-Hydroinformatics/HydroErr",
    keywords=[
        "hydrology",
        "error",
        "metrics",
        "comparison",
        "statistics",
        "forecast",
        "observed",
    ],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    install_requires=[
        "numpy",
        "scipy",
    ],
)
