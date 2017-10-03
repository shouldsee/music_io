#!/usr/bin/env python
import util
import os
import numpy as np
dname = 'sample'
# fname = 'sample'
flst = os.listdir(dname)
# bname = flst[0]
for bname in flst:

	fname = os.path.join(*[dname,bname])
	# print fname
	p = util.piece( fname, alias = 'comp_80')
	p.set_wavelet('morlet')
	# util.piece

	p.plot(0,0.05, scale = np.arange(1,80))