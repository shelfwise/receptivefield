from setuptools import setup

setup(
  name = 'receptivefield',
  packages = ['receptivefield'],  
  version = '0.1.1',
  description = 'Gradient based Receptive field estimation library',
  author = 'fornax.ai',
  author_email = 'krzysztof.kolasinski@fornax.ai',
  url = 'https://github.com/fornaxai/receptivefield',
  download_url = 'https://github.com/fornaxai/receptivefield/archive/0.1.tar.gz', 
  keywords = ['tensorflow', 'keras', 'receptivefield'],
  install_requires=[
     'matplotlib', 'pillow', 'numpy', 'keras'
  ],
  classifiers = [],
)
