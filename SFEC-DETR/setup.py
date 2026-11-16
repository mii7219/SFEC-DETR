from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent

setup(
    name='ultralytics',
    version='8.0.201',
    description='M7 object detection training and inference',
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    python_requires='>=3.8,<3.12',
    license='AGPL-3.0',
    license_files=['LICENSE', 'NOTICE'],
    packages=find_packages(include=['ultralytics', 'ultralytics.*']),
    package_data={'ultralytics': ['cfg/*.yaml']},
    install_requires=(ROOT / 'requirements.txt').read_text().splitlines(),
)
