from setuptools import setup
import sys, os

# Most of the relevant info is stored in this file
info_file = os.path.join('twix', 'info.py')
exec(open(info_file).read())


setup(name=NAME,
      description=DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      classifiers=CLASSIFIERS,
      platforms=PLATFORMS,
      version=VERSION,
      provides=PROVIDES,
      packages=['twix'],
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRES,
     )

