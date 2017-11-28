from setuptools import setup

VERSION = '0.2.1'

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
        'matplotlib', 'pillow', 'numpy', 'keras'
    ],
    classifiers=[],
    include_package_data=True
)
