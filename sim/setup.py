from setuptools import setup, find_packages

setup(
    name='arm4r-sim',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    license=open('LICENSE').read(),
    zip_safe=False,
    description="RLBench Sim for ARM4R",
    author='Dantong Niu',
    author_email='niudantong.88@gmail.com',
    url='n/a',
    install_requires=[line for line in open('requirements.txt').readlines() if "@" not in line],
    keywords=['Transformer', 'Behavior-Cloning', 'Langauge', 'Robotics', 'Manipulation'],
)
