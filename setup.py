import subprocess
from setuptools import setup, find_packages


# Get install requirements
with open('requirements.txt', 'r') as f:
    install_requires = list()
    dependency_links = list()
    for line in f:
        re = line.strip()
        if re:
            install_requires.append(re)

setup(name='ccfs-python',
      version='0.2.0',
      description='CCFs are a decision tree ensemble method for classification and regression',
      author='UBC',
      url='https://github.com/plai-group/ccfs-python.git',
      maintainer_email='tonyjos@cs.ubc.ca',
      maintainer='Tony Joseph',
      license='Apache-2.0',
      packages=find_packages(exclude=['datasets', 'dataset']),
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=install_requires,
      keywords='canonical-correlation-forests'
      )
