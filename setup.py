# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('LICENSE') as f:
    license = f.read()

with open('README.md') as f:
    readme = f.read()

setup(
    name='dataint',
    version='0.0.1',
    description='Python Client for the Data Integration Project',
    long_description=readme,
    author='au.csiro.data61',
    url='https://github.com/NICTA/data-integration-py',
    license=license,
    packages=find_packages(exclude=('tests'))
)
