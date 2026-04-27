from glob import glob
from setuptools import find_packages, setup

package_name = 'tb3_nav2_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/maps', glob('maps/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Johnny Vang',
    maintainer_email='johnnyjvang@gmail.com',
    description='TurtleBot3 Nav2 + SLAM experiments in Gazebo',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'single_goal_nav = tb3_nav2_slam.single_goal_nav:main',
            'single_goal_return = tb3_nav2_slam.single_goal_return:main',
            'goal_from_list = tb3_nav2_slam.goal_from_list:main',
            'random_safe_goal_explorer = tb3_nav2_slam.random_safe_goal_explorer:main',
            'timer_based_patrol_explorer = tb3_nav2_slam.timer_based_patrol_explorer:main',
        ],
    },
)