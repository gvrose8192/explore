from setuptools import find_packages, setup

package_name = 'explore'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Greg',
    maintainer_email='gvrose8192@gmail.com',
    description='Lidar explore with obstacle detection',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'explore_node = explore.explore_node:main'
        ],
    },
)
