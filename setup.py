from os import path

from setuptools import find_packages, setup


here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

pip_packages = [
    'gym==0.26.2',
    'gymnasium==0.28.1',
    'noise==1.2.2',
    'numba==0.60.0',
    'numpy==1.26.4',
    'opencv-python>=4.8.0',
    'pyglet==1.5.23',
    'sample-factory>=2.1.1',
    'scipy==1.14.1',
    'six>=1.16.0',
    'torch==2.5.0',
]

setup(
    name='swarm_rl',
    version='1.0.0',
    description='Single-quadrotor lidar obstacle traversal reinforcement learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(where='.'),
    python_requires='>=3.11',
    install_requires=pip_packages,
)
