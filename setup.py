from setuptools import setup, find_packages

setup(
    name='Sibyl',
    version='0.4',
    packages=find_packages(),
    install_requires=['cython', 'numpy', 'quandl', 'matplotlib', 'pandas', 'sklearn', 'tensorflow-macos']
)