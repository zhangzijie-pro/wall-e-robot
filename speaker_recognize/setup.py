from setuptools import setup, find_packages

setup(
    name='speaker_recognition',
    version='0.1.0',
    description='A speaker recognition project using PyTorch, torchaudio, and librosa.',
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
        'tqdm>=4.66.0',
        'torchvision>=0.16.0',
        'pydub>=0.25.0',
        'redis',
        'MNN',
        "onnx"
    ],
    python_requires='>=3.10',
)
