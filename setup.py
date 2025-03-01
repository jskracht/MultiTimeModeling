from setuptools import setup, find_packages

setup(
    name='Sibyl',
    version='0.5',
    packages=find_packages(),
    install_requires=['cython', 'fredapi', 'numpy', 'matplotlib', 'pandas', 'scikit-learn', 'tensorflow-macos']
)