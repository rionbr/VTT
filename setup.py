from setuptools import setup
from vtt import __version__, __release__

def readme():
	with open('README.md') as f:
		return f.read()

setup(
	name='VTT',
	version=__version__,
	description="Variable Trigonometric Threshold (VTT)",
	long_description="A machine learning linear classifier as defined in A. Abi-Haidar, J. Kaur, A. Maguitman, P. Radivojac, A. Retchsteiner, K. Verspoor, Z. Wang, and L.M. Rocha [2008]. \"Uncovering protein interaction in abstracts and text using a novel linear model and word proximity networks\". Genome Biology. 9(Suppl 2):S11 .\nTo be used in conjunction with the Scikit-Learn python package.",
	classifiers=[
		'Development Status :: 4 - Beta',
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 2.7',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Information Analysis',
	],
	keywords="machine learning linear model computational biology",
	url="http://github.com/rionbr/vtt",
	author="Rion Brattig Correia",
	author_email="rionbr@gmail.com",
	license="MIT",
	packages=['vtt'],
	install_requires=[
		'numpy',
		'scipy',
		'sklearn',
	],
	include_package_data=True,
	zip_safe=False,
	)