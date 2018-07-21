from setuptools import setup

VERSION = '0.4.0'

setup(
    name='receptivefield',
    packages=['receptivefield'],
    version=VERSION,
    description='Gradient based Receptive field estimation library',
    author='fornax.ai',
    author_email='krzysztof.kolasinski@fornax.ai',
    url='https://github.com/fornaxai/receptivefield',
    download_url=f'https://github.com/fornaxai/receptivefield/archive/{VERSION}.tar.gz',
    keywords=['tensorflow', 'keras'],
    install_requires=[
        'pillow>=4.1.1',
        'matplotlib>=2.0.2',
        'numpy>=1.14.3',
    ],
    classifiers=[],
    include_package_data=True
)
