import os
from setuptools import setup, find_packages
import setuptools


setup(
    name="game24scaling",
    version="0.1.0",
    description="Exploring Scaling Laws in Language Model Tasks: The Game of 24",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    author="Angel Raychev, Yalcin Tur, Mihajlo Stojkovic",
    author_email="angelray@stanford.edu, yalcintr@stanford.edu, mstojkov@stanford.edu",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    keywords="language models, scaling laws, game of 24, tree search, stanford",
    python_requires='>=3.10',
)