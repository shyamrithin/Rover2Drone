# =============================================================================
# File:        src/rover2drone_nav/setup.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-24
# =============================================================================
"""ament_python setup for rover2drone_nav (route_follower, world_overlay)."""
from setuptools import find_packages, setup

package_name = 'rover2drone_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shyam',
    maintainer_email='shyamrithin44@gmail.com',
    description='Rover2Drone route follower and RViz world overlay',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'route_follower = rover2drone_nav.route_follower:main',
            'world_overlay = rover2drone_nav.world_overlay:main',
        ],
    },
)
