from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = 	[
					Extension("cython_utils", ["cython_utils.pyx"],
								include_dirs=[numpy.get_include()],
								extra_compile_args=["-Ofast"]
					)
				]

setup(
	name = 'utils',
	cmdclass = {'build_ext': build_ext},
	ext_modules = ext_modules
)
