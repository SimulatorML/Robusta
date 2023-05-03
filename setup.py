import setuptools
from setuptools import setup
import os

cwd = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(cwd, 'requirements.txt')

with open(file) as f:
    dependencies = list(map(lambda x: x.replace("\n", ""), f.readlines()))

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='robusta',
    author='Bogdan Gromov',
    author_email="uberkinder@yandex.com",
    description="Robusta ML Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=dependencies,
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Science',

        'License :: OSI Approved :: MIT License',

        'Natural Language :: English',

        'Operating System :: OS Independent',

        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',

        'Topic :: Data Science :: Machine Learning',
    ]
)
