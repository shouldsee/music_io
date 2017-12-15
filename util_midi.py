
####### MIDI utilities

import mido
import numpy as np
import matplotlib.pyplot as plt
import mir_eval

import StringIO

persistent = (
    range(17,24+1)   ### Organs
    + range(41,45+1) ### Strings (persistent)
#    + range(49,55+1) ### Ensemble (persis)
    + range(57,64+1) ### Brass
    + range(65,80+1) ### Reed and Pipe
)


def smpte2second(msg):
    fps = msg.frame_rate
#     fpsInSec = 1./fps
    InSec = msg.hours * 3600 + msg.minutes * 60 + msg.seconds + (msg.frames + msg.sub_frames / 10.) / fps
    return InSec

def track2midi(track, sample_dt = 0.05, persistent_only = 1,ticks_per_beat = None,TEMPO = None,stdout = StringIO.StringIO(),
               n_sample = 10,THRESHOLD = 0.5, v_base = 0.5, offset_insec = 0.0,**kwargs):
        
    signal_0 = [0]*128
    sample_intick = mido.second2tick( sample_dt/n_sample,ticks_per_beat,TEMPO)
    offset_intick = mido.second2tick( offset_insec,ticks_per_beat,TEMPO)
    #print type(sample_in_tick)
    LENGTH = int( (time_track(track) + offset_intick) // sample_intick) + 1 
    OUTPUT = [signal_0 ]* LENGTH

#     track.ONOFF = detect_format(track)
    it = (x for x in track)
    
    isMETA = 1
    #INST   = None
    while True:
        msg = next(it,None)
        if msg is None:
            omsg = "[WARN]:No notes were detected in %s" % track[0]
            print omsg
            return None
        if msg.type=='program_change':
            INST = msg.program + 1
#             raise Exception( )
#     for msg in it:  
        if isMETA:        
            if msg.type =="note_on":
#                assert INST  is not None, "No instrument detected"
                #if INST is None:
                    #return None
                #if persistent_only:
#                    if not INST in persistent:
#                        return None
                #    assert INST  in persistent, "Instrument %d is not persistent"%INST
                assert TEMPO is not None, "Cannot determine tempo"
                print >>stdout, msg
    #             print msg.time
                isMETA=0
                break

#    t_curr = 0.
#    t_curr = offset_intick + sample_intick / 2
#    t_curr = offset_intick 
    t_curr = -offset_intick 
    #t_curr = -offset_intick + sample_intick
#    t_curr = -offset_intick
#    t_track= 0.
    t_track= msg.time 
    
    signal = signal_0
    signal_new = receive(signal_0,msg)
    print >>stdout,"============"



    for i in range(len(OUTPUT)):
#         if not isMETA:
#     #         mtype = msg.type
#             if track.ONOFF == 'OnOnly':
#                 if msg.type == "note_on" and msg.velocity==0:
#                     msg.__dict__['type'] = "note_off"                
#        print >>stdout,sum(signal)
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
    if OUTPUT is None:
        return OUTPUT
    OUTPUT = OUTPUT[:len(OUTPUT)//n_sample * n_sample]
    DIM = OUTPUT.shape
    DIM = (-1,n_sample,DIM[-1])
    OUTPUT = OUTPUT.reshape(DIM).mean(axis = 1)
    OUTPUT[OUTPUT<(THRESHOLD * v_base)] = 0 
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


def receive(signal,msg,use_velocity = 1):
    signal = signal[:]
    if msg.type=='note_on':
        if msg.velocity ==0:
            signal[msg.note]=0.
        else:
            if use_velocity:
                signal[msg.note]=msg.velocity/128.
            else:
                signal[msg.note]=1
    elif msg.type=='note_off':
        signal[msg.note]=0.
    else:
#         print "[WARN]'%s' msg is not recognised" % msg
        assert 0,"[ERROR]'%s' msg is not recognised" % msg
    return signal
def get_tempo(mid,as_message= 0):
    for track in mid.tracks:
        for msg in track:    
            if msg.type =="set_tempo":
                TEMPO=msg if as_message else msg.tempo                
                return TEMPO
            if msg.time != 0:
                break
    raise Exception("Cannot find tempo for %s" % mid)
def get_smpte(mid,as_message= 1):
    for track in mid.tracks:
        for msg in track:    
            if msg.type =="smpte_offset":
                SMPTE = msg if as_message else msg               
                return SMPTE
            if msg.time != 0:
                break
    raise Exception("Cannot find smpte_offset for %s" % mid)

    
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

def extract_midi_roll(filename, sample_dt = 0.05, DEBUG = True, persistent_only = 1,**kwargs):
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
    mid.offset_insec = smpte2second(get_smpte(mid)) - 0.05 ##### This might relate to piece.to_chunk(20) with 20*0.05 = 1
    lst = []
    try:
        for track in mid.tracks:
            mroll = track2midi(track, sample_dt = sample_dt, TEMPO=mid.TEMPO,ticks_per_beat= mid.ticks_per_beat,
                              persistent_only = persistent_only,
                               offset_insec = mid.offset_insec,
                              **kwargs)
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

def norm_by_rmsq(chunks, norm = 1):
    SUM = np.sqrt(np.mean( np.power(chunks,2),axis = 1,keepdims=1))
    if norm:
        SUM[SUM==0]=1
        OUT = chunks/SUM
    else:
        OUT = SUM
    #     OUT = np.nan_to_num(OUT)
    return OUT

def mroll2chroma(mroll):
#     if chroma:
    mroll = mroll[:,12:120]
    SP = list(mroll.shape)
#         SP[1] = 12; SP.append(-1)
#         mroll = np.reshape(mroll,SP).sum(axis = 2)
    SP[1] = -1; SP.append(12)
    cmroll = np.reshape(mroll,SP).sum(axis = 1)
#     SUM = cmroll.sum(axis = 0,keepdims =1 )
#     SUM[SUM==0]=1
#     cmroll = cmroll / SUM.astype(float)
    return cmroll 

def midi_roll_play(mroll,chroma = False):
    if mroll.shape[-1]==12:
        freqs = mir_eval.transcription.util.midi_to_hz(np.arange(60,72))
    else:
        freqs = mir_eval.transcription.util.midi_to_hz(np.arange(0,128))
#     mroll = np.log(mroll+1E-5)
    Xs_exp = mir_eval.sonify.time_frequency( mroll.T,freqs,times = 1./20*np.arange(len(mroll)),fs = 16000)
#     Xs_exp = np.hstack([Xs_exp,[0]*(len(cpXs)-len(Xs_exp))])
    return Xs_exp/np.sqrt((Xs_exp**2).mean())
if __name__=='__main__':
    fname = 'sample/MIDI/composer-bach-edition-bg-genre-cant-work-0002-format-midi1-multi-zip-number-01.mid'
    extract_midi_roll(fname)
    
    
import util
def mroll_feed(fname,norm = False, truncate = 10, DEBUG = 0,
               offset = 2, #### If coded correctly this should be 0, but ATM set to 3 to compensate for any bug
              ):
    assert fname.endswith('.mroll.npy'),'Must be .mroll.npy file'
    bname = fname.rstrip('.mroll.npy')

    mroll = np.load(fname)    
    p = util.piece( '%s.wav' % bname)
    # print p.x0.dtype
#     p.downsample(16000)

    if p.x0.dtype =='int16':
        p.x0 = p.x0.astype("float32") /2**15
    elif p.x0.dtype == 'float32':
        pass
    else:
        raise Exception("wavfile coded in %s"%p.x0.dtype)
        
    chunks = np.array(p.to_chunk(20))
    if DEBUG:
        print mroll.shape,chunks.shape
    ldiff = len(mroll) - len(chunks)
    assert ldiff in [-1,0,1],"Shape of .mroll.npy and .wav does not match: %s %s" %(mroll.shape,chunks.shape)
    #### This might be a bug, where transcribed wav is sometimes longer than midi roll
    ##### UPDATE: using FluidSynth instead of Timidity fixed this BUG!!
    chunks = chunks[offset:]
    idx = min(len(mroll),len(chunks))
    chunks = chunks[:idx];mroll = mroll[:idx]
    if norm:
        #### Normalise chunks by sumsq
    #             chunks = util_midi.norm_by_rmsq(chunks)
    #         NZidx = list(np.nonzero(np.sum(chunks,axis = 1)))

        eps = 1E-10
#         # Gaussian norm
#         chunks_SD = chunks.std(axis = 1,keepdims= 1) + eps
#         chunks_M = chunks.mean(axis = 1,keepdims= 1)
#         NZidx = np.squeeze( (chunks_SD)/chunks_M > 8)
#         chunks = (chunks - chunks_M)/chunks_SD
#         chunks = chunks[NZidx,:]; mroll=mroll[NZidx,:]            
#         chunks = (chunks - chunks_M[NZidx,:])/chunks_SD[NZidx,:]

    #             #### RMSQ filter
    #             RMSQ = util_midi.norm_by_rmsq(chunks, norm = 0)
    #             NZidx = np.squeeze( RMSQ> 1E-4)
    #             chunks = chunks[NZidx,:]; mroll=mroll[NZidx,:]            
    #             assert len(NZidx)>0,"No valid chunks"
    #             assert chunks.shape[0]>0,"No valid chunks"

        ### RMSQ filter and norm
        RMSQ = norm_by_rmsq(chunks, norm = 0) + eps
        NZidx = np.squeeze( RMSQ> 1E-4)
        chunks = chunks/RMSQ
        chunks = chunks[NZidx,:]; mroll=mroll[NZidx,:]            
#         chunks = chunks + eps

#         assert len(NZidx)>0,"No valid chunks"
    assert chunks.shape[0] - 1 > truncate ,"No valid chunks"
    chunks = chunks[truncate:]; mroll = mroll[truncate:]
    chunks = np.expand_dims(chunks,0)
    #         d = {
    #             "name":bname,
    #             "sound":chunks,
    #             "mroll":mroll}.items()
    #         pk.dump(d, open('%s.npy' % bname,'wb') )
    d = [chunks,mroll]
    #         d = [bname,chunks,mroll]
    np.save('%s.both.npy' % bname, d)
#             "name":bname,
#             "sound":chunks,
#             "mroll":mroll}
# #             bname,[chunks,mroll]}
#                )
    return 1
    
   