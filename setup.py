import os
from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

class ThirdLibcpp(install):
    def run(self):
        subprocess.run([
            'python3', 'Tools/lib_build.py',
            '--yaml', 'Tools/libs.yaml',
            '--output', 'include'
        ], check=True)

        subprocess.run([
            'python3', 'Tools/model.py',
            '--yaml', 'Tools/model_config.yaml'
        ], check=True)

        super().run()


setup(
    name='walle-robot',
    version='0.1.0',
    description='A robot framework for Wall-E',
    long_description=open('README.md').read(),  
    author='zhangzijie',
    author_email='zzj01262022@163.com',
    packages=find_packages(),
    install_requires=[
        'torch>=2.1.0',
        'torchaudio>=2.1.0',
        'librosa>=0.10.1',
        'numpy>=1.26.0',
        'scipy>=1.11.0',
        'matplotlib>=3.8.0',
        'soundfile>=0.12.1',
        'torchvision>=0.16.0',
        'pydub>=0.25.0',
        'redis',
        'MNN',
        "onnx",
        "opencv-python",
        "Pillow",
        "gradio_imageslider",
        "gradio==4.29.0",
        "ChatTTS",
        "vosk",
    ],
    python_requires='>=3.10',
    cmdclass={
        'install': ThirdLibcpp,
    }
)
