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
# from types import NoneType
NoneType = type(None)
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
        self.xs = x0  
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
    def trimto(self, t1 = None, t2 = None):
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
        self.xs = xs
        self.ts = ts
        return [ts,xs]
    def downsample(self, bitrate):
        idx = np.arange(0,len(self.t0), self.bitrate/bitrate).astype('int')
        self.t0 = self.t0[idx]
        self.x0 = self.x0[idx]
        self.bitrate = bitrate
    def cwt(self, t1,t2,  scale = None, xs = None, postFunc =  lambda x:x, p = 20):       
        # if isinstance(postFunc, None):
        #     postFunc = lambda x:x
        if isinstance(scale, type(None)):
            scale = np.arange(1,129)       
        self.scale = scale
        # self.freqs = 1. / self.scale * p
        self.freqs = 1. / self.scale * p
        # self.freqs = self.bitrate / self.scale

#         self.scale = np.arange(1,100)
        # tmax = 0.1

        if isinstance( xs, type(None)):
            ts,xs = self.trimto(t1,t2)
        # ts,xs = self.trimto(t1,t2)
            
#         xs = xs/ np.std(xs)
#         self.ts

        # wavelet = pywt.ContinuousWavelet( self.motherwave, )
        # wavelet.center_frequency = 1
        # coef, freqs = pywt.cwt( xs, scale,  wavelet,
        #     # sampling_period = self.bitrate,
        #     sampling_period = 1./self.bitrate,
        #     )

        # coef = scipy.signal.cwt(xs, getattr(scipy.signal,self.motherwave),
        # widths = self.scale,  )
        coef = mlpywt.cwt( xs, 1./self.bitrate, self.scale, wf = self.motherwave, p = p )
        # coef = mlpywt.cwt( xs, 1, self.scale, wf = self.motherwave, p = p )
        coef = postFunc(coef)
        freqs = None
        self.coef = coef
        # self.freqs= freqs
        return coef,freqs
    def icwt(self, ts = None , coef = None, scale = None, p = 20):
        if isinstance(scale, type(None)):
            scale = self.scale
#         if isinstance(coef, type(None)):
#             coef = self.coef
        if isinstance( ts, type(None)):
            ts = self.ts
        xs = mlpywt.icwt( np.real(coef), 1./self.bitrate, scale, wf = self.motherwave, p = p  )
        return [ts,xs]        

    def old_cwt(self, t1, t2):
#         tmin = int(self.bitrate * t1)
#         tmax = int(self.bitrate * t2)
#         x = self.t0[tmin:tmax]
#         y = self.x0[tmin:tmax]
        x, y = self.trimto(t1,t2)
        scale = np.arange(1,129)
        # scale = np.linspace(1,1000,100)
        coef, freqs=pywt.cwt( y, scale, self.motherwave)
        self.coef = coef
        return coef
#         scale = np.arange(1,128)
#         x, y = self.trimto(t1,t2)
#         mlpywt.cwt(self.x0, self.t0[1]-self.t0[0], scale, wf='morlet', )
    def set_pdir(self, pdir = None):
        if isinstance(pdir, type(None)):
            pdir = 'gallery/'
        assert os.path.isdir(pdir)

        self.pdir = pdir        
    def plot(self,t1,t2, alias = None, lineonly = 0, show = 1, save = 1,
            coef = None,
            log = 1,
            dpi = 300,
            big = 0,
            ofreqs = None,
             **kwargs):
        if alias:
            self.alias = alias  
        if not getattr(self, 'pdir'):
            self.set_pdir()
        if not isinstance(ofreqs, NoneType):
            freqs = ofreqs
        else:
            freqs = self.freqs
        # scale = np.arange(1,257)
        # if isinstance(pdir, type(None)):
        #     self.pdir = 'gallery/'

        self.bfname = os.path.basename(self.fname).split('.')[0]
        title = '%s_%s' % ( self.bfname, self.alias) 
        tmin = int(self.bitrate * t1)
        tmax = int(self.bitrate * t2)
        x = self.t0[tmin:tmax]
        y = self.x0[tmin:tmax]


        if not lineonly:
            if isinstance( coef, type(None)):
                coef, _ = self.cwt( t1,t2 , **kwargs)
#                 coef, freqs=pywt.cwt( y, scale, self.motherwave)
            
                self.coef = coef
        # ax2.matshow(coef) 
            if log:
                im = np.log10( 1 + 10*abs(coef))
            else:
                im = coef
            # f_im = ax2.pcolormesh( x, self.freqs * self.bitrate , im)
            # f_im = ax2.pcolormesh( x, np.log10(self.freqs)  , im)
            # ax3.plot( 10 * np.mean(im, axis = 1)  , self.scale)
            # ax3.set_xlim(right = 0.05)
#             ax2.pcolormesh( x, self.scale, np.log10( abs(coef)))
        
#             ax2.pcolormesh( x, self.scale, np.log10(1 + 10*abs(coef)))
#             ax2.imshow( np.log10(1 + 10 * abs(coef)),)
#         extent = [min(x),max(x),min(self.scale),max(self.scale),]
        # ax2.imshow( x, scale, coef)
#         plt.tight_layout()

        fig = plt.figure(figsize = [10,10])
        if big:
            ax2 = plt.subplot(111)
        else:
            ax1 = plt.subplot(311)
            ax2 = plt.subplot(312)        
            ax3 = plt.subplot(313)
            ax1.plot(x,y)
            ax1.set_xlim([min(x),max(x)])

        f_im = ax2.pcolormesh( x, freqs  , im) 
        ax2.set_yscale('log')
        fig.colorbar(f_im)
        try:
            ax3.plot(  np.log10(self.scale), np.log10(freqs) ,)
            ax1.set_title(title)
        except:
            print "doing big plot"
        if save:
#             plt.savefig('gallery/' + title + '.png' , )
            fig.savefig( self.pdir + title + '.png', dpi = dpi )
        if not show:
            fig.set_visible(False)
        plt.close(fig)
        # else

#             fig.close()
#             plt.hide(fig)
#             plt.show() 
    def play(self,t1 = None, t2 = None):
        if t1 is None or t2 is None:
            xs = self.xs
        else:
            ts,xs = self.trimto( t1, t2)

        obj = ipd.Audio( xs , rate = self.bitrate)     
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
    
    def downsample(self,new_rate):
        ratio = self.bitrate/new_rate
        idx = np.arange(0, len(self.t0) ,ratio).astype(int)
        self.x0 = self.x0[idx]
        self.xs = self.x0[:]
        self.t0 = np.arange(0,len(idx)) * 1./new_rate
        self.bitrate = new_rate 

    def to_chunk(self,new_rate):
        ratio = self.bitrate/new_rate
        idx = np.arange(0, len(self.t0) ,ratio).astype(int)
        print len(idx)
        return np.array(np.split(self.x0, idx[1:])[:-1])


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
        


####### MIDI utilities

import mido
import numpy as np
import matplotlib.pyplot as plt


import StringIO
def track2midi(track, sample_dt = 0.05,ticks_per_beat = None,TEMPO = None,stdout = StringIO.StringIO(),**kwargs):
    signal_0 = [0]*128
    sample_intick = mido.second2tick( sample_dt,ticks_per_beat,TEMPO)
    LENGTH = int(time_track(track) // sample_intick) + 1 
    OUTPUT = [signal_0 ]* LENGTH

#     track.ONOFF = detect_format(track)
    it = (x for x in track)
    
    isMETA = 1
    while True:
        msg = next(it,None)
        if msg is None:
            omsg = "[WARN]:No notes were detected in %s" % track[0]
            print omsg
            return None
#             raise Exception( )
#     for msg in it:  
        if isMETA:        
            if msg.type =="note_on":
                assert TEMPO is not None, "Cannot determine tempo"
                print >>stdout, msg
    #             print msg.time
                isMETA=0
                break

    t_curr = 0.
    t_track= msg.time
    
    signal = receive(signal_0,msg)
    signal_new = signal
    print >>stdout,"============"



    for i in range(len(OUTPUT)):
#         if not isMETA:
#     #         mtype = msg.type
#             if track.ONOFF == 'OnOnly':
#                 if msg.type == "note_on" and msg.velocity==0:
#                     msg.__dict__['type'] = "note_off"                
        print >>stdout,sum(signal)
        OUTPUT[i] = signal[:]
        t_curr += sample_intick
    #     t_new = t_curr + sample_intick    
    #     while t_new > t_track and msg:
        print >>stdout,t_curr,t_track
        print >>stdout,(t_curr > t_track)
    #     print (t_curr > t_track) & (msg is not None)
        if t_curr > t_track:
            signal = signal_new
#             signal_new = signal

        flipped = False
        while (t_curr > t_track) & (msg is not None):
            msg = next(it, msg)
            print >> stdout, msg
            if msg.type =='end_of_track':
                break
#             if not msg:
#                 break
            t_track += msg.time
            if t_curr > t_track:
                signal = receive(signal,msg)
            else:
                if not flipped:
                    signal_new = signal
                    flipped = True
                signal_new = receive(signal,msg)
            print >>stdout,t_curr,t_track
    OUTPUT = np.array(OUTPUT)
    return OUTPUT     

if __name__=='__test__':
    mroll = track2midi(track,TEMPO=mid.TEMPO,ticks_per_beat= mid.ticks_per_beat)     
    plot_midi_roll(mroll)

def time_track(track):
    return( sum(msg.time for msg in track))
# def time_
# time_track(track)
# for track in mid.tracks:
#     track.time = time_track(track)
# MAX_TICK = max(t.time for t in mid.tracks)


def receive(signal,msg):
    signal = signal[:]
    if msg.type=='note_on':
        if msg.velocity ==0:
            signal[msg.note]=0
        else:
            signal[msg.note]=1
    elif msg.type=='note_off':
        signal[msg.note]=0
    else:
#         print "[WARN]'%s' msg is not recognised" % msg
        assert 0,"[ERROR]'%s' msg is not recognised" % msg
    return signal
def get_tempo(mid):
    for track in mid.tracks:
        for msg in track:    
            if msg.type =="set_tempo":
                TEMPO=msg.tempo
                return TEMPO
            if msg.time != 0:
                break
    raise Exception("Cannot find tempo for %s" % mid)

    
def midi_merge(*args):
    for e in args:
        assert e.shape[-1] == 128," Make sure shape fits"
    if not args:
        return None
    
    args = [x for x in args if len(x) > 10]    
    LEN = max(len(x) for x in args)
    args = [np.lib.pad( x, ((0,LEN - len(x)),(0,0)), 'constant', constant_values=[0.]) for x in args]
    return np.vstack([x[None,:] for x in args]).max(axis = 0)

def plot_midi_roll(mroll, **kwargs):
    plt.figure(figsize = [15,6],**kwargs)
    if len(mroll)!=128:
        mroll = mroll.T
    plt.pcolormesh(mroll)
    
    
# plot_midi_roll(mroll)

def extract_midi_roll(filename, sample_dt = 0.05, DEBUG = True):
    '''
    Sample a midi file into a 128-bit stream at given rate. 
    Input:
        filename: path to midi file
        sample_dt: interval in seconds

    Return:
        Numpy array of the shape ( time, 128 )  (midi encodes 128 pitches)
        NoneType if failed
    '''
    mid = mido.MidiFile(filename)
    mid.TEMPO = get_tempo(mid)
    lst = []
    try:
        for track in mid.tracks:
            mroll = track2midi(track, sample_dt = sample_dt, TEMPO=mid.TEMPO,ticks_per_beat= mid.ticks_per_beat)
            if mroll is not None:
        #         plot_midi_roll(mroll)
                lst.append(mroll)
        OUTPUT = midi_merge(*lst)
        if DEBUG:
            plot_midi_roll(OUTPUT)
        return OUTPUT
    except Exception as e:
        print e
        return None

def norm_by_rmsq(chunks):
    OUT = chunks/np.sqrt(np.mean( np.power(chunks,2),axis = 1,keepdims=1))
#     OUT = np.nan_to_num(OUT)
    return OUT

if __name__=='__main__':
    fname = 'sample/MIDI/composer-bach-edition-bg-genre-cant-work-0002-format-midi1-multi-zip-number-01.mid'
    extract_midi_roll(fname)