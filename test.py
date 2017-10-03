#!/usr/bin/env python
import util
import os
import numpy as np
import scipy
dname = 'sample'
# fname = 'sample'
flst = os.listdir(dname)
# bname = flst[0]
for bname in flst:

	fname = os.path.join(*[dname,bname])
	# print fname
	ali = 'valleys'
	# thres = 0.05
	# ali = 'valleys_%d' % thres

	# ali = 'comp_256'
	p = util.piece( fname, alias = ali)
	# p.set_wavelet('morlet')
	# p.set_wavelet('morl')
	p.set_wavelet('cmor')
	# util.piece

	# p.plot(0,0.05, scale = np.arange(1,80))
	# p.plot(0,0.05, scale = np.arange(1,129))
	scale = np.arange(25,175)
	coef = p.cwt( 0, 0.05, scale )
	# pks  = util.detect_peaks( - np.log10( 1 + 10*abs(coef)), ry = 20 )

	im = np.log10( 1 + 10*abs(coef))
	# im = scipy.ndimage.gaussian_filter(im, [1,.5])
	im = scipy.ndimage.filters.laplace(im)
	im = scipy.ndimage.gaussian_filter(im, [2,1])
	im = np.log(1 + np.maximum(im,0))
	# c1 = c1 > th
	# ec1 = (np.power(10,c1) - 1 )/10
	p.plot(0,0.05, scale = scale
		, coef = im
		, log  = 0
		# , coef = coef
		)