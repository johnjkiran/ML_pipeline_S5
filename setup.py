from setuptools import setup, find_packages

setup(
    name="mnist_classifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'pytest>=7.0.0',
        'tqdm>=4.64.0',
    ],
) 