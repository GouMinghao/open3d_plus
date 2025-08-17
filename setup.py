from distutils.core import setup
from setuptools import find_packages
from open3d_plus.version import __version__

setup(
    name="open3d_plus",
    version=__version__,
    description="open3d plus functions",
    author="Minghao Gou",
    author_email="gouminghao@gmail.com",
    url="",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "open3d>=0.8.0.0",
        "opencv-python",
    ],
)
