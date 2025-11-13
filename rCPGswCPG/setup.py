from setuptools import setup, find_packages

setup(
    name='rCPGswCPG',
    version='0.1.0',
    author='Pavel Tolmachev',
    author_email='betadecay1993@gmaill.com',
    description='A package for modeling rCPG and swCPG interactions.',
    long_description_content_type='text/markdown',
    url='https://github.com/ptolmachev/SimplifiedModel_rCPG_swCPG_interaction',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        # Add other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)