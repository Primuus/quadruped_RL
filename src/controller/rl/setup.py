from setuptools import find_packages, setup

package_name = 'rl_controller'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        'setuptools',
        'numpy',
        'pyyaml',
        'pyserial>=3.5',
        'evdev',
        'torch',
        'mujoco>=3.2.3',
        'onnxruntime>=1.16,<2',
    ],
    python_requires='>=3.10',
    zip_safe=True,
    maintainer='RY',
    maintainer_email='18737950912@163.com',
    description='Quadruped RL deployment package for direct Python MuJoCo simulation.',
    license='Apache-2.0',
)
