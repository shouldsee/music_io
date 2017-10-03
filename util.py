import scipy.signal
import scipy.ndimage
import scipy.io.wavfile
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os
import IPython.display as ipd
from IPython.display import display_html as HTML
# import mlpy.wavelet as mlpywt

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
    def save(self, fname):
        f_abs = os.path.join( self.dirname, fname)
        scipy.io.wavfile.write( f_abs, self.bitrate, self.x0, )
        self.fname = f_abs

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
        wavelet = pywt.ContinuousWavelet( self.motherwave, )
        wavelet.center_frequency = 1
        coef, freqs = pywt.cwt( xs, scale,  wavelet,
            sampling_period = self.bitrate,
            # sampling_period = 1./self.bitrate,
            )
        # coef = scipy.signal.cwt(xs, getattr(scipy.signal,self.motherwave),
        # widths = self.scale,  )
        # coef = mlpywt.cwt( xs, 1, self.scale, wf = self.motherwave, p=2 )
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
            coef = None,
            log = 1,
             **kwargs):
        if alias:
            self.alias = alias  
        # scale = np.arange(1,257)
        pdir = 'gallery/'

        self.bfname = os.path.basename(self.fname).split('.')[0]
        title = '%s_%s' % ( self.bfname, self.alias) 
        tmin = int(self.bitrate * t1)
        tmax = int(self.bitrate * t2)
        x = self.t0[tmin:tmax]
        y = self.x0[tmin:tmax]


        fig = plt.figure(figsize = [10,10])
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)        
        ax3 = plt.subplot(313)
        ax1.plot(x,y)
        ax1.set_xlim([min(x),max(x)])
        if not lineonly:
            if isinstance( coef, type(None)):
                coef = self.cwt( t1,t2 , **kwargs)
#                 coef, freqs=pywt.cwt( y, scale, self.motherwave)
            
                self.coef = coef
        # ax2.matshow(coef) 
            if log:
                im = np.log10( 1 + 10*abs(coef))
            else:
                im = coef
            f_im = ax2.pcolormesh( x, self.scale, im)
            fig.colorbar(f_im)
            ax3.plot( 10 * np.mean(im, axis = 1)  , self.scale)
            ax3.set_xlim(right = 0.05)
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


from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

def detect_peaks(image, rx = 2, ry = 10  ):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(rx,ry)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, 
        # size = (rx,ry),
        footprint=neighborhood
        )==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks
        