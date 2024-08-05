import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='sprout-ml',
     version='0.3',
     scripts=[],
     author="Tommaso Zoppi",
     author_email="tommaso.zoppi@unifi.it",
     description="SPROUT - a Safety wraPper thROugh ensembles of UncertainTy measures",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/tommyippoz/SPROUT",
     keywords=['machine learning', 'safety wrapper', 'safety monitor', 'uncertainty measures', 'ensemble'],
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )