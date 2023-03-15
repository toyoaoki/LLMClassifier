from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='llmclassifier',
    version='0.1.0',
    packages=['llmclassifier'],
    install_requires=requirements,
)