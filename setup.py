from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='stick_slip_learn',
      version='0.1',
      description='Training algorithms to predict stick and slip in granular avalanches',
      long_description=readme(),
      classifiers=[
        'Development Status :: 0 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Applied Machine Learning:: Granular Avalanches',
      ],
      url='http://github.com/lenkatilka/stick_slip_learn',
      author='Lenka Kovalcinova',
      author_email='lenka.kovalcinova@gmail.com',
      license='MIT',
      packages=['stick_slip_learn'],
      install_requires=['markdown'],
      zip_safe=False)
