from setuptools import setup

setup(
    name='dev-diary',
    version='0.1.0',
    author='sommd',
    description='A small (one file) command line tool for keeping a developer diary.',
    license='MIT',
    url='https://github.com/sommd/dev-diary',
    py_modules=['diary'],
    install_requires=[
        'click>=6.0',
        'sqlalchemy>=1.1.0',
        'python-dateutil>=2.6.0'
    ],
    entry_points={
        'console_scripts': [
            'diary=diary:diary'
        ]
    }
)
