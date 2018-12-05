#python3 setup_cython.py  build_ext --inplace
import numpy as np
cimport numpy as np
from numpy import empty_like, empty
from libc.math cimport sin
#from libc.stdlib cimport rand, RAND_MAX
import cython
cdef float PI = np.pi
DTYPE = np.float
ctypedef np.float_t DTYPE_t
cdef int LENGTH_OF_DAY = 24*60


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calcDemandCython(np.ndarray[np.long_t, ndim=1] t, float t0, float maxD, float minD):
	cdef int n = t.shape[0]
	cdef double factor = (maxD-minD)*0.5
	cdef int i
	cdef np.ndarray[np.float64_t, ndim=1] ret = empty_like(t,dtype=np.float)
	for i in range(n):
		ret[i] = factor * (-sin(2*PI*(t[i]-t0)/LENGTH_OF_DAY)+1) + minD
	return ret

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calcProfiles(np.ndarray[np.long_t, ndim=1] t, int size, int offset, float maxD, float minD):
	cdef int n = t.shape[0]
	cdef double factor = (maxD-minD)*0.5
	cdef double tmp = 0.0
	cdef int i,j
	cdef np.ndarray[np.float64_t, ndim=2] ret = empty((size,n),dtype=np.float)
	
	for i in range(n):
		tmp = factor * (-sin(2*PI*(t[i])/LENGTH_OF_DAY)+1) + minD
		for j in range(size):
			#idx = i + j
			#ret[j,idx] = tmp
			ret[j,(i+j*offset)%n] = tmp
	return ret

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calcDemandF(np.ndarray[np.long_t, ndim=1] t, float t0, float maxD, float minD):
	cdef double factor = -(maxD-minD)*0.5
	cdef np.ndarray[np.float64_t, ndim=1] ret = empty_like(t,dtype=np.float)
	for i in range(LENGTH_OF_DAY):
		ret[i] = factor * (sin(2*PI*(t[i]-t0)/LENGTH_OF_DAY)+1) + minD
	return ret

@cython.boundscheck(False)
def randomWalk(int length, double P = 1.0, double v0 = 0.004, double rh = 0.02):
	cdef np.ndarray[np.float64_t, ndim = 1] chi = np.random.normal(scale=1,size=length)
	cdef np.ndarray[np.float64_t, ndim = 1] pricewalk = np.empty(length,dtype=np.float)
	
	pricewalk[0] = P
	cdef int idx
	for idx in range(1,length):
		pricewalk[idx]= -v0*(pricewalk[idx-1]-P)+rh*chi[idx-1]+pricewalk[idx-1]

	return pricewalk
	