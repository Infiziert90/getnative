#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    install_requires = fh.read()

setup(
    name="getnative",
    version='2.2.0',
    description='Find the native resolution(s) of upscaled material (mostly anime)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Infi, Kageru',
    author_email='infiziert@protonmail.ch, kageru@encode.moe',
    url='https://github.com/Infiziert90/getnative',
    install_requires=install_requires,
    python_requires='>=3.6',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': ['getnative=getnative.app:main'],
    }
)
