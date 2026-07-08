from setuptools import find_packages
from distutils.core import setup

setup(name='quadruped_rl',
      version='1.0.0',
      author='Unitree Robotics',
      license="BSD-3-Clause",
      packages=find_packages(),
      author_email='support@unitree.com',
      description='Quadruped reinforcement learning training workspace',
      install_requires=['isaacgym', 'matplotlib', 'numpy==1.20', 'tensorboard', 'pyyaml'])
