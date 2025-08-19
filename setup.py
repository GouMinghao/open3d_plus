from distutils.core import setup
from setuptools import find_packages

setup(
    name="open3d_plus",
    version="0.3.2",
    description="open3d plus functions",
    author="Minghao Gou",
    author_email="gouminghao@gmail.com",
    url="",
    packages=find_packages(),
    install_requires=["numpy", "open3d", "opencv-python", "matplotlib"],
)
