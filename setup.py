from setuptools import setup, find_packages

setup(
    name='rfp',
    version='0.1.0',
    description='Reguralized Bi-level Gradient Descent',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/pharringtonp19/rfp',  # URL to the repository
    author='Patrick Power',
    author_email='your.email@example.com',
    license='MIT',  # Or whatever license you choose
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here, e.g.,
        # 'numpy',
        # 'pandas',
    ]
)