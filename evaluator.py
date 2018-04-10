from pymisca.util import *
from pymisca.vis_util import *
import IPython.display as ipd
import matplotlib as mpl
import mir_eval
def midi_roll_play(mroll,chroma = False):
    if mroll.shape[-1]==12:
        freqs = mir_eval.transcription.util.midi_to_hz(np.arange(60,72))
    elif mroll.shape[-1]==128:
        freqs = mir_eval.transcription.util.midi_to_hz(np.arange(0,128))
#    elif mroll.shape[-1]==36:
    else:
        bfreq = 48
        ufreq = bfreq + mroll.shape[-1]
        freqs = mir_eval.transcription.util.midi_to_hz(np.arange(bfreq,ufreq))
#     mroll = np.log(mroll+1E-5)
    SHAPE = mroll.shape
    mroll = mroll.T
#     mroll = mroll * freqs.T[:,None]
#     mroll = norm_by_freq(mroll)
    Xs_exp = mir_eval.sonify.time_frequency( mroll, freqs,times = 1./20*np.arange(SHAPE[0]),fs = 16000)
#     Xs_exp = np.hstack([Xs_exp,[0]*(len(cpXs)-len(Xs_exp))])
    return Xs_exp/np.sqrt((Xs_exp**2).mean())
def to_chunk(self,new_rate):
    ratio = self.bitrate/new_rate
    idx = np.arange(0, len(self.ts) ,ratio).astype(int)
    print len(idx)
    return np.array(np.split(self.xs, idx[1:])[:-1])

def transcribe(signal, model, norm = 1, chroma = 1, log = 1):
    if isinstance( signal, util.piece):
        signal.downsample(16000)
        signal = signal.to_chunk(20)
#     plt.figure()

#     assert not (pXs[0] - Xs[0]).any()
#     pXs = Xs[:500]
#     pYs_exp = Ys[:500]
    pYs_act = model.predict_on_batch(signal) 
#     pYs_act = pYs_act * np.arange(pYs_act.shape[1])[None,:]
    pYs_act += 1E-10
#     compare(log = 1)
    if chroma:
#        pYs_exp = mroll2chroma(pYs_exp)
        pYs_act = mroll2chroma(pYs_act,norm = 1)        
    Z1 = pYs_act.T
    if log:
        plt.pcolormesh(Z1,
                       alpha = 1.0,
                  norm=mpl.colors.LogNorm(vmin=Z1.min(), vmax=Z1.max())
                  )
    else:
        plt.pcolormesh(Z1)
    fs = np.arange(pYs_act.shape[-1])[None,:]
    pYs_act = (pYs_act)*fs
    ipd.display(ipd.Audio(midi_roll_play(pYs_act),rate = 16001.))
    return pYs_act
import librosa
import librosa.display

def stft(p):
    sr = p.bitrate
    y  = p.xs
    chroma_cq = librosa.feature.chroma_stft(y=y, sr=sr,
                                           n_chroma=12, n_fft=800,
                                           hop_length=800)
    librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time')
    plt.show()
    return chroma_cq.T# cqt(p)
# stft(p)
# compare(log = 0)

import util#

# best_agent = agent
def cqt(p):
    sr = p.bitrate
    y  = p.xs
    chroma_cq = librosa.feature.spectral.cqt(y=y, sr=sr, 
#                                            window = 50,
#                                           hop_length = 1
                                          )
    librosa.display.specshow(chroma_cq, y_axis='cqt_note', x_axis='time',
#                             y_coords=np.arange(0,84)+1
                            )
    return chroma_cq.T
def evaluate( model,wavfile = 'sample/waltz_for_toutzy.wav',log= 1,YLIM = [0,48]):
# log = 1
# if 1:
    if hasattr(model,'model'):
        model = model.model
    
#     wavfile = 'sample/waltz_for_toutzy_50.wav'
#     wavfile = 'sample/Tamacun.wav'
#     wavfile = 'sample/MIDI/composer-bach-edition-bg-genre-cant-work-0002-format-midi1-multi-zip-number-01.wav'
    p = util.piece(wavfile)
    print p.x0.max()
    if p.x0.dtype=='int16':
        p.x0 = p.x0.astype('float32')
        p.x0 = p.x0/2**15
    p.xs = p.x0
    # p.xs = p.xs.astype('float32')
    p.downsample(16000)
    # p.bitrate = 18000
#     p.trimto(18,26)
#     p.trimto(28,40)
    p.trimto(50,66)
#     p.trimto(60,100)
    # p.trimto(100,140)
    # print len(p.xs)
    chunks = to_chunk(p,20)[:]
    chunks = np.array(chunks)
    # chunks = util_midi.norm_by_rmsq(chunks,norm = 1)

    eps = 1E-8
    plt.figure(figsize = [12,6])
    mroll = transcribe(chunks, model,chroma = 0,log = log)
    
    ytk = librosa.midi_to_note(range(0,128))
    plt.yticks(np.arange(0,128) +.5,librosa.midi_to_note(range(0,128)))
#     plt.ylim(40,78)
    plt.grid()
#     plt.ylim(0,36)
    plt.ylim(YLIM)

    ipd.display(ipd.Audio(p.xs,rate=16000))
    plt.figure(figsize = [12,6])
    # print p.xs.dtype
    cqt(p)
#     plt.yscale('log')
    plt.yticks(np.exp(np.linspace(*np.log(plt.gca().get_ylim()),num=48)),librosa.midi_to_note(np.arange(48)+24))
    plt.show()
    return mroll