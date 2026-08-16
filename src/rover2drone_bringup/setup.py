"""
setup.py
========

Package configuration for rover2drone_bringup.

Beyond the ament_python defaults, this installs the launch/ directory into
the package share directory so `ros2 launch rover2drone_bringup <file>` can
find it. Without the extra data_files entry, launch files are built but
never installed, and ros2 launch reports the file as not found.
"""

import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'rover2drone_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shyamrithin',
    maintainer_email='shyamrithin44@gmail.com',
    description='Bringup launch files for the Rover2Drone marsupial '
                'wind turbine inspection stack.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)