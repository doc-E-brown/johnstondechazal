#!/usr/bin/env python
"""The setup script."""

import versioneer
from setuptools import find_packages, setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy>=1.18.1',
    'pandas>=1.0.1',
    'versioneer>=0.18',
    'tqdm>=4.43.0',
    'Click>=7.0',
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest>=3',
]

setup(
    author="Ben Johnston",
    author_email='ben.johnston@sydney.edu.au',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description=
    "A method of optimising facial landmark selection from crowd sourced labels.",
    entry_points={
        'console_scripts': [
            'johnstondechazal=johnstondechazal.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='johnstondechazal',
    name='johnstondechazal',
    packages=find_packages(include=['johnstondechazal', 'johnstondechazal.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/docEbrown/johnstondechazal',
    #    version=versioneer.get_version(),
    version='0.0.0',
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
