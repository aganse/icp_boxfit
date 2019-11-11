from setuptools import setup

setup(
    name='icp_boxfit',
    version='0.1',
    description='ICP-based 3D box fit to point cloud data',
    author='Andy Ganse',
    author_email='andy@ganse.org',
    install_requires=[
        'icp',
        'matplotlib',
        'numpy',
        'scikit-learn',
    ],
)
