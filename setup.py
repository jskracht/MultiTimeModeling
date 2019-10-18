from setuptools import setup, find_packages

setup(
    name='Sibyl',
    version='0.1',
    packages=find_packages(),
    install_requires=['quandl', 'statsmodels', 'pycausalimpact']
)