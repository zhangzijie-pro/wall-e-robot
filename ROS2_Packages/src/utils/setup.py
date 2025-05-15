from setuptools import setup

package_name = 'utils'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_dir={'': 'shared_utils'},
    install_requires=['setuptools'],
    zip_safe=True,
    author='zzj',
    author_email='zzj01262022@163.com',
    description='Shared utilities and custom messages/services',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [],
    },
)
