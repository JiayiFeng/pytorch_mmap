from setuptools import setup, find_packages

setup(
    name="pytorch_mmap",
    version="0.1.0",
    install_requires=[
        "torch"
    ],
    packages=find_packages(),
)