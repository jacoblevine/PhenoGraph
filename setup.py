from setuptools import setup, find_packages

setup(
    name="PhenoGraph",
    description="Graph-based clustering for high-dimensional single-cell data",
    version="1.0",
    author="Jacob Levine",
    author_email="jl3545@columbia.edu",
    packages=find_packages(),
    url="https://github.com/jacoblevine/PhenoGraph",
    license="LICENSE.md",
    long_description=open("README.md").read(),
    install_requires=open("requirements.txt").read()
)