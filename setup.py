from setuptools import setup, find_packages

setup(name='crosswalk_reproduction',
      version='0.1.0',
      description='Reproduction of the Crosswalk paper',
      long_description=open('README.md', encoding="utf8").read(),
      url='',
      author='DDC Person',
      author_email='info@example.com',
      license='',
      packages=find_packages(),
      zip_safe=False,
      install_requires=[],
      entry_points={
            'console_scripts': ['xwalk_reprod=crosswalk_reproduction.__main__:main']
      }
      )
