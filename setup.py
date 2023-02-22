from setuptools import setup

setup(name='tesi_phd_slm',
      version='',
      description="Repository for my PhD project, concerning the use of Meadowlark's Spatial Light Modulator in an adaptive optics test-bed",
      url='',
      author='edoardo',
      author_email='',
      license='MIT',
      packages=['tesi_slm'],
      install_requires=[
          'plico_dm_server',
          'plico_dm',
          'plico',
          'pysilico',
          'numpy',
          'arte',
          'astropy'
      ],
      package_data={
          'tesi_slm': ['data/*'],
      },
      include_package_data=True,
      )
 