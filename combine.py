import util

dname = 'sample'
p1 = util.piece(
	'violin-c-hi-long.wav' 
	,dirname = dname)
p2 = util.piece(
	'violin-g-low-long.wav' 
	,dirname = dname)
# fname = 'sample'
p3 = util.piece(
	'violin-g-low-long.wav' 
	,dirname = dname)
tmax = min(p1.t0[-1],p2.t0[-1])
_,x1 = p1.trimto(0, tmax)
_,x2 = p2.trimto(0, tmax)
# xs, ts = 
p3.x0 = x1 + x2
p3.save('violin-cg-long.wav')