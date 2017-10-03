import scipy.signal
import scipy.ndimage
import scipy.io.wavfile
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os
import IPython.display as ipd
from IPython.display import display_html as HTML
import mlpy.wavelet as mlpywt

class piece(object):
    def __init__(self, fname, alias = 'test',dirname = '', ):
        self.dirname = ''
        if dirname:
            self.setdir(dirname)
        self.load(fname)
        self.alias = alias

    def setdir(self, dirname):
        assert os.path.isdir(dirname)
        self.dirname = dirname
    def load(self, fname ):
        f_abs = os.path.join( self.dirname, fname)
        assert os.path.isfile(f_abs)
        bitrate, mat = scipy.io.wavfile.read(f_abs)
        self.fname = f_abs
        if mat.ndim > 1:
            x0 = mat[:,0]
        else:
            x0 = mat
#         self.meta = {'bitrate':bitrate}
        self.bitrate = bitrate
        self.x0 = x0
        self.t0 = np.arange(0,x0.size)/float(bitrate)        
    def set_wavelet(self,  motherwave):
        self.motherwave = motherwave
    def swt(self,t1 ,t2, level):
        tmin = int(self.bitrate * t1)
        tmax = int(self.bitrate * t2)
        x = self.t0[tmin:tmax]
        y = self.x0[tmin:tmax]
        scale = np.arange(1,129)
        coef = pywt.swt( y, self.motherwave, level,)
        return coef
#         self.coef = coef
#         return coef
    def trimto(self,t1 = None,t2 = None):
        if not t1:
            tmin = 0
        else:
            tmin = int(self.bitrate * t1)            
        if not t2:
            tmax = len(self.t0)
        else:
            tmax = int(self.bitrate * t2)
#         tmin = int(self.bitrate * t1)
#         tmax = int(self.bitrate * t2)
        ts = self.t0[tmin:tmax]
        xs = self.x0[tmin:tmax]
        return [ts,xs]
    def cwt(self, t1,t2, scale = None, postFunc =  lambda x:x):       
        # if isinstance(postFunc, None):
        #     postFunc = lambda x:x
        if isinstance(scale, type(None)):
            scale = np.arange(1,129)
        self.scale = scale
#         self.scale = np.arange(1,100)
        tmax = 0.1
        ts,xs = self.trimto(t1,t2)
#         xs = xs/ np.std(xs)
#         self.ts
        coef = mlpywt.cwt( xs, 1, self.scale, wf = self.motherwave, p=2 )
        coef = postFunc(coef)
        self.coef = coef
        return coef
    def icwt(self, ts = None , coef = None, scale = None):
        if isinstance(scale, type(None)):
            scale = self.scale
#         if isinstance(coef, type(None)):
#             coef = self.coef
        if isinstance( ts, type(None)):
            ts = self.ts
        xs = mlpywt.icwt( np.real(coef), 1, scale, wf = self.motherwave, p=2 )
        return [ts,xs]        

    def old_cwt(self, t1, t2):
#         tmin = int(self.bitrate * t1)
#         tmax = int(self.bitrate * t2)
#         x = self.t0[tmin:tmax]
#         y = self.x0[tmin:tmax]
        x, y = self.trimto(t1,t2)
        scale = np.arange(1,129)
        coef, freqs=pywt.cwt( y, scale, self.motherwave)
        self.coef = coef
        return coef
#         scale = np.arange(1,128)
#         x, y = self.trimto(t1,t2)
#         mlpywt.cwt(self.x0, self.t0[1]-self.t0[0], scale, wf='morlet', )        
    def plot(self,t1,t2, alias = None, lineonly = 0, show = 1, save = 1,
            coef = None, **kwargs):
        if alias:
            self.alias = alias  
        scale = np.arange(1,257)
        pdir = 'gallery/'

        self.bfname = os.path.basename(self.fname).split('.')[0]
        title = '%s_%s' % ( self.bfname, self.alias) 
        tmin = int(self.bitrate * t1)
        tmax = int(self.bitrate * t2)
        x = self.t0[tmin:tmax]
        y = self.x0[tmin:tmax]


        fig = plt.figure(figsize = [10,10])
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        ax1.plot(x,y)
        ax1.set_xlim([min(x),max(x)])
        if not lineonly:
            if isinstance( coef, type(None)):
                coef = self.cwt( t1,t2 , **kwargs)
#                 coef, freqs=pywt.cwt( y, scale, self.motherwave)
            
                self.coef = coef
        # ax2.matshow(coef) 
            ax2.pcolormesh( x, self.scale, np.log10( 1 + 10*abs(coef)))
#             ax2.pcolormesh( x, self.scale, np.log10( abs(coef)))
        
#             ax2.pcolormesh( x, self.scale, np.log10(1 + 10*abs(coef)))
#             ax2.imshow( np.log10(1 + 10 * abs(coef)),)
#         extent = [min(x),max(x),min(self.scale),max(self.scale),]
        # ax2.imshow( x, scale, coef)
#         plt.tight_layout()
        ax1.set_title(title)
        if save:
#             plt.savefig('gallery/' + title + '.png' , )
            fig.savefig( pdir + title + '.png', dpi = 300)
        if not show:
            fig.set_visible(False)
            plt.close(fig)
#             fig.close()
#             plt.hide(fig)
#             plt.show() 
    def play(self,t1 = None, t2 = None):
        ts,xs = self.trimto(t1,t2)       
        obj = ipd.Audio(xs, rate = self.bitrate)     
        return HTML(obj)
    def extract(self,t1,t2):
        ts, xs =p.trimto( t1, t2 )
        coef = p.cwt( t1,t2)
        im = np.log10(1 + 10 * abs(coef))
        im = scipy.ndimage.gaussian_filter(im, .5)
        c1 = scipy.ndimage.filters.laplace(im)
        imb = scipy.ndimage.gaussian_filter(im, 10)
        imb = scipy.ndimage.maximum_filter(imb, size = [30,200] )
        imb = scipy.ndimage.maximum_filter(imb, size = [10,100] )
        c2 = imb < np.mean(imb,0) +  1. * np.std(imb,0)

        bl = c1 > np.expand_dims(np.mean(c1,axis = 0) + 1.0 * np.std(c1,axis = 0), axis = 0)
#         bl = c1 > np.mean(c1,axis = 0) + .5 * np.std(c1,axis = 0)
        ccoef = coef.copy()
#         ccoef[:] = 0 
#         ccoef[c2] = 1
#         ccoef[bl] = 0
        ccoef = (1 - bl) * .5
        ccoef[c2] = 0
        return ccoef

    def reconstruct(self, t1, t2):
        ts, xs =p.trimto( t1, t2 )
        coef = p.cwt( t1,t2)
        im = np.log10(1 + 10 * abs(coef))
        im = scipy.ndimage.gaussian_filter(im, .5)
        c1 = scipy.ndimage.filters.laplace(im)
        imb = scipy.ndimage.gaussian_filter(im, 10)
        imb = scipy.ndimage.maximum_filter(imb, size = [30,200] )
        imb = scipy.ndimage.maximum_filter(imb, size = [10,100] )
        c2 = imb < np.mean(imb,0) +  1. * np.std(imb,0)

        bl = c1 > np.expand_dims(np.mean(c1,axis = 0) + 1.0 * np.std(c1,axis = 0), axis = 0)
#         bl = c1 > np.mean(c1,axis = 0) + .5 * np.std(c1,axis = 0)
        ccoef = coef.copy()
#         ccoef[:] = 0 
#         ccoef[c2] = 1
#         ccoef[bl] = 0
        ccoef = (1 - bl) * .5
        ccoef[c2] = 0
#         ccoef[bl] = 0
#         ccoef[:1,:] = 0
        
        ts ,xs = p.icwt( ts, coef = ccoef )
        return [ts,xs]


        