# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py
import os, sys
from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements

setup(
    name='eblib',
    version='0.0.1',
    description='Machine Learning package for Anomaly Detection',
    long_description=readme,
    author='Fumito Ebuchi',
    author_email='fumito.ebuchi@gmail.com',
    url='',
    license=license,
    install_requires=read_requirements(),
    packages=find_packages(exclude=('tests', 'docs'))
)

