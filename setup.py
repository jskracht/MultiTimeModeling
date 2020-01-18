from setuptools import setup, find_packages

setup(
    name='Sibyl',
    version='0.2',
    packages=find_packages(),
    install_requires=['quandl', 'matplotlib', 'tensorflow', 'pandas', 'pycausalimpact']
)