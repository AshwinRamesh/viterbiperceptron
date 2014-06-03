from setuptools import setup

readme = open('README.md').read()
setup(name='ViterbiPerceptron',
      version='0.1',
      author='Ashwin Ramesh',
      author_email='ashramesh1992@gmail.com',
      description='Averaged Structured Perceptron (Viterbi) for POS/NER Tagging',
      long_description=readme,
      setup_requires=[],
      install_requires=[],
      test_suite='nose.collector',
      entry_points={
        'console_scripts': []
      },
      scripts=[],
      packages=['viterbiperceptron'])
