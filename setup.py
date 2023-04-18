from setuptools import find_packages
from setuptools import setup

setup(
    name="dustpylib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "dustpy", "numpy", "scipy",
    ],
)