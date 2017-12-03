#!/usr/bin/env python
import util
import os
import numpy as np
import scipy
dname = 'sample2'
# fname = 'sample'
flst = os.listdir(dname)
# bname = flst[0]
for bname in flst:

	fname = os.path.join(*[dname,bname])
	# print fname
	# ali = 'valleys'
	# ali = 'freqs_abs'
	omega = 8
	def pf(x):
		return np.real(x)
	wname = 'morlet'
	dt = 500
	itn = 60
	i = 0
	ali = '%s_%dms_f%d_real_iter%d'% ( wname, dt, omega, i)
	# ali = 'freqs_500ms_f20_real'
	# ali = 'freqs_500ms_comp'

	# thres = 0.05
	# ali = 'valleys_%d' % thres

	# ali = 'comp_256'
	p = util.piece( fname, alias = ali)
	p.set_wavelet('morlet')
	p.set_wavelet('morlet')
	p.set_pdir('gallery2a/')
	# p.set_wavelet('morl')
	# p.set_wavelet('cmor')

	# p.downsample(10000)

	# util.piece



	# p.plot(0,0.05, scale = np.arange(1,80))
	# p.plot(0,0.05, scale = np.arange(1,129))
	# scale = np.arange(25,175)

	# freqs = p.bitrate * np.power(10,np.linspace(1.5,4.3,60))
	# scale = p.bitrate / freqs 
	# omega = 10

	logfmax = np.log10( p.bitrate / omega )
	logfmin = np.log10( 20. / omega )

	freqs = np.power(10, np.linspace( logfmin, logfmax, 200))
	scale = 1./ freqs
	# scale = np.linspace(15,500,100)
	tmin = 10.
	tmax = tmin + dt/1000.
	# tmax = 11.
	# tmax = 17.5
	coef, _ = p.cwt( tmin, tmax, scale, p = omega, postFunc = pf)


	im = np.log10( 1 + 10*abs(coef))
	p.plot( tmin, tmax, scale = scale
	, coef = im
	, log  = 0 
	, big = 1
	# , dpi = 500
	, dpi = 400
	# , coef = coef
	)	

	for i in range(1,itn+1):
		lst = []
		for ri,si in zip(coef,scale):
			# print si
			# engsq1 = (np.sum(np.square(ri)))
			engsq1 = (np.mean(abs((ri))))
			# eng1 = np.sqrt(np.sum(np.square(ri)))
			# eng1 = np.sqrt(np.mean(np.square(ri.astype('float'))))
			row, _ = p.cwt(tmin,tmax,
					scale = np.array([si]*2,), 
					xs= ri,
					p = omega, 
					postFunc = pf,)
			# engsq2 = (np.sum(np.square(row[0])))
			engsq2 = (np.mean(abs(row[0])))
			# print engsq1,engsq2
			# eng2 = np.sqrt(np.sum(np.square(row[0])))
			# lst.append(row[0] * np.sqrt(engsq1 / float(engsq2)))
			lst.append(row[0] * float(engsq1) / float(engsq2))
		coef = lst[:] 
		coef = np.array(coef)
		p.alias = '%s_%dms_f%d_real_iter%d'% ( wname, dt, omega, i)
		print len(scale)
		im = np.log10( 1 + 10*abs(coef))
		p.plot( tmin, tmax, scale = scale,
			ofreqs = freqs * omega,
		coef = im,
		log  = 0 ,
		big = 1,
		dpi = 400,
		)	
	# pks  = util.detect_peaks( - np.log10( 1 + 10*abs(coef)), ry = 20 )

	# coef = np.real(coef)
	im = np.log10( 1 + 10*abs(coef))
	

	# im = abs(coef)

	# im = scipy.ndimage.filters.laplace(im)
	# im = scipy.ndimage.gaussian_filter(im, [2,1])
	# im = np.log(1 + np.maximum(im,0))

	# c1 = c1 > th
	# ec1 = (np.power(10,c1) - 1 )/10

	# p.plot( tmin, tmax, scale = scale
	# 	, coef = im
	# 	, log  = 0 
	# 	, big = 1
	# 	# , dpi = 500
	# 	, dpi = 400
	# 	# , coef = coef
	# 	)