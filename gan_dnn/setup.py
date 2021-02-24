from setuptools import setup

setup(
       # the name must match the folder name 
        name="gan_dnn", 
        version= '0.1',
        author="Alex",
        author_email="pas_trop2003@yahoo.com",
        description='Sequence GAN',
        packages=['gan_dnn'],
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['GAN'],
        classifiers= [
            "Development Status :: 1 - Planning",
            "Intended Audience :: All",
            "Programming Language :: Python :: 3",
        ]
)