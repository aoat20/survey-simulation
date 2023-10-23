import setuptools

setuptools.setup(
    name='survey-simulation',
    version='0.12',
    author='Alfie Anthony Treloar',
    author_email='aoat20@bath.ac.uk',
    description='Simple survey simulation',
    url='https://github.com/aoat20/survey-simulation',
    license='Not yet licensed', 
    packages=['survey_simulation'],
    install_requires=['opencv-python', 
                      'matplotlib', 
                      'numpy'],
)
