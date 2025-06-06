from setuptools import find_packages, setup

package_name = 'face_voice_id_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,              # Installs face_voice_id_node/
        f'{package_name}.script',  # Installs script/
        f'{package_name}.script.Face',  # Installs script/Face/
        f'{package_name}.script.embedding'  # Installs script/embedding/
        ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zzj',
    maintainer_email='zzj01262022@163.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'face_voice_id_node=face_voice_id_node.main_node:main',
        ],
    },
)
