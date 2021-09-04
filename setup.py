"""Module installation script.

Installation
------------
Install module 'gym_marl_reconnaissance' as

    $ cd gym-marl-reconnaissance/
    $ pip install -e .

"""
from setuptools import setup

setup(name='gym_marl_reconnaissance',
      version='0.0.1',
      install_requires=[
        'numpy',
        'Pillow',
        'matplotlib',
        'gym',
        'pybullet',
        'stable_baselines3'
        ]
)
