from util_midi import persistent
from pymisca.util import *
import random
import mido
import os,util_midi
import numpy as np

def clean_this_shit(mid,DEBUG = 0,rand_inst = [62,63,64],pitch_shift = 0,pitch_center = 0,ini_offset = 100):
    keep = [
#         'time_signature',
        'track_name','end_of_track','note_on','note_off']
    notes = ['note_on','note_off']
    tracks = []
    newmid = mido.MidiFile(type = 1)
    TEMPO = None
    TSig = None
#     SMPTE = None 
#    SMPTE = mido.MetaMessage(type = 'smpte_offset',frame_rate=25,hours=0,minutes=0,seconds=0,frames=1,sub_frames=0,time=0)
    SMPTE = mido.MetaMessage(type = 'smpte_offset',frame_rate=25,hours=0,minutes=0,seconds=0,frames=0,sub_frames=0,time=0)
    for (i,track) in enumerate(mid.tracks):
        COUNT = 0
        INST  = None
        Didx = []
        newtrack = mido.MidiTrack()
        META = 0
#         CHANNEL = None
        for (i,msg) in enumerate(track):
            if msg.type in keep:
                if msg.type in notes:
                    COUNT += 1
                    if pitch_center:
                        msg.note = 2 * pitch_center - msg.note
                    msg.note = msg.note + pitch_shift
                    if msg.velocity != 0:
                        msg.velocity = 64
#                     assert INST is not None,' INST is None \n %s' %msg
#                     CHANNEL = INST.channel
#                     assert isinstance(INST.channel,int),"CHANNEL is %s" % CHANNEL
                    msg.channel  = INST.channel
                    if COUNT == 1:
                        msg.time = msg.time + ini_offset

                newtrack.append(msg)
            else:
                if TSig is None:
                    if msg.type =='time_signature':
                        TSig = msg
                        newtrack.append(msg)
                if TEMPO is None:
                    if msg.type =="set_tempo":                                
                        TEMPO=msg.tempo                
                        newtrack.append(msg)
                        newtrack.append(SMPTE)
                        META = 1
                if INST is None:
                    if msg.type =='program_change':
    #                     INST = msg.program
                        INST = msg
                        if rand_inst:
                            INST.program = random.choice(rand_inst)-1
#                         if CHANNEL is None:
#                             CHANNEL = msg.channel
                        newtrack.append(msg)

        if DEBUG:
            print len(newtrack)
        if (not META) & (COUNT==0):   #### trackname, inst, end
#         if (newtrack[0].type =='track_name') and (newtrack[-1].type =='end_of_track'):
            #### discard this track
            pass
        else:
            tracks.append(newtrack[:])
    newmid.tracks = tracks
    if DEBUG:
        print newmid
    assert len(newmid.tracks) != 0, "No tracks left after cleaning"
    return newmid
# mid
if __name__=='__main__':
#     mid = mido.midifiles.MidiFile(filename='test.mid')
    filename = 'sample/MIDI/composer-bach-edition-bg-genre-cant-work-0002-format-midi1-multi-zip-number-04.mid'
    mid = mido.MidiFile(filename)
    clean_this_shit(mid).print_tracks()
                # get_tempo(mid).print_tracks()
    

def single_tempo(mid,as_message= 0,DEBUG = 0):
    TEMPO = None
    for track in mid.tracks:
        Didx = []
        for (i,msg) in enumerate(track):    
            if msg.type =="set_tempo":                
                if TEMPO is None:
                    TEMPO=msg if as_message else msg.tempo                
                else:
                    Didx.append(i)
#                 print msg
#                 return TEMPO
            elif msg.type in ['smpte_offset','control_change']:
                Didx.append(i)
            
            elif msg.time != 0:
                break
        for x in Didx[::-1]:
            _ = track.pop(x )
            if DEBUG:
                print _
    if TEMPO is None:
        raise Exception("Cannot find tempo for %s" % mid)
    return (mid)
def filter_on_INST(mid, permit = None,DEBUG=0, default_INST = 43):
    if permit is None:
        permit = persistent
        
    Didx = []
    for (i,track) in enumerate(mid.tracks):
        metaEND = None
        INST = None
        CHANNEL = None
        for (j,msg) in enumerate(track):
            if metaEND is None:
                if msg.type in ['note_on','note_off','program_change']:
                    metaEND = j
#             else:
                    if CHANNEL is None:                    
#                         if hasattr(msg,"channel"):
                        CHANNEL = msg.channel
            if msg.type =="program_change":
                INST = (msg.program + 1)
                if INST not in permit:
                    Didx.append(i)
                    break
        ##### Set program to default
        if INST is None and metaEND is not None and CHANNEL is not None:
            msg = mido.Message(type='program_change',channel = CHANNEL , program = default_INST - 1, time = 0)            
            track.insert(metaEND,msg)
            mid.tracks[i] = track
    for x in Didx[::-1]:
        _ = mid.tracks.pop(x)
        if DEBUG:
            print _        
    return mid

def clean(mid):

    for (i,track) in enumerate(mid.tracks):
        Didx = []
        for (j,msg) in enumerate(track):    
            if msg.type =="smtpe":
                if TEMPO is None:
                    TEMPO=msg if as_message else msg.tempo                
                else:
                    Didx.append(i)
#                 print msg
#                 return TEMPO
            elif msg.time != 0:
                break
        for x in Didx[::-1]:
            _ = track.pop(x )
            if DEBUG:
                print _

def check_midi(fname,**kwargs):
    assert fname.endswith('mid'), "Must pass a '.mid' file"

    mroll = util_midi.extract_midi_roll(fname, DEBUG=0,**kwargs)
#         SUM = mroll.sum(axis = 1,keepdims = 1)
#         SUM[SUM==0]=1
#         mroll = mroll/SUM
    if mroll is None:
        raise Exception("mroll is None")
    else:
        assert mroll.shape[-1]==128,"make sure mroll is coded in midi"
        
    bname = fname.rsplit('.',1)[0]
#     tim_buffer = timidify(fname)
    tim_buffer = fluidify(fname)
    mroll = np.array(mroll)
    np.save('%s.mroll' % bname, mroll)
    return 1

# 3447570/16000.
def soxify(fname,sr = 16000):
#     tempname = '%s.mid'%fname
#     _ = !cp {fname} {tempname}
    outname = '%s.wav' % fname.rsplit('.',1)[0]
    cmd = u'sox -t raw -r {sr} -e signed -b 16 -c 1 {fname} {outname}'
    res = get_ipython().getoutput(cmd)
#     !rm {tempname}
    return res
# soxify(mid.filename)
def fluidify(fname,sr = 16000,
                SFfile = '/usr/share/sounds/sf2/FluidR3_GM.sf2'):
    
    outname = '%s.wav' % fname.rsplit('.',1)[0]
    cmd = u'fluidsynth -r {sr} -F {outname} {SFfile} {fname}'
    res = get_ipython().getoutput(cmd)
    assert not res.grep('No preset found on*'), "MIDI contains missing preset"
#     res = !fluidsynth {fname}
    return res
# fluidify(mid.filename)
def timidify(fname, sr = 16000):
    cmd = u'timidity  -Ow -s {sr} {fname}'
    res = get_ipython().getoutput(cmd)
    assert not OUT.grep('No instrument mapped to*'), "MIDI file not recognised"
    return res

# %%time


# suc = map(midi_feed,FILES)


def func_DIR_mid2single(fname,**kwargs):
    try:
        pitch_shift = kwargs.get('pitch_shift',0)
        pitch_center= kwargs.get('pitch_center',0)
        mid =  mido.MidiFile(fname)
#         mid =  single_tempo(mid,DEBUG = 0)
        mid = filter_on_INST(mid)
        mid = clean_this_shit(mid,**kwargs)
#         if mid.tracks is []:
#             return 0
#         if mid.tracks:
#             mid = filter_on_INST(mid,DEBUG = 1)
#         print mid
        newname = fname.rstrip('.mid') + 'P%sS%s.single_mid'%(pitch_shift,pitch_center)
        print newname
        mid.save(newname)
        return 1
    except Exception as e:
        print '%s\n%s' %(e , fname)
        return 0
def DIR_mid2single(DIR, cap = 50,para = 12,clean = 1,**kwargs):       
    FILES0 = list(os.walk(DIR))[0][-1]
    FILES0 = [os.path.join(DIR,f) for f in FILES0]
#     for f in FILEs:
#         os.remove(f)
    if clean:
        for f in FILES0:
            if f.endswith('.single_mid'):
                if os.path.isfile(f):
    #         fname = f+'.single_mid'
                    os.remove(f)
    #### scan all midi
    FILES = [f.rstrip('.mid') for f in FILES0 if f.endswith(".mid")]
    FILES = [f+'.mid' for f in FILES]
    FILES = FILES[:cap]
    func = functools.partial(func_DIR_mid2single,**kwargs)
    suc = mp_map(func,FILES,para)
#     suc = mp_map(func,FILES,1)    
    print "All:",len(suc)
    print "skipped:",sum(1 for x in suc if x is None)
    print "success:",sum(x for x in suc if x is not None)
    print "Done"
# spwave.read(fname)

# func = functools.partial(midi_feed,norm = 1)
# func = functools.partial(midi_feed,norm = 0)
# func = check_midi
def func_DIR_single2mroll(fname):
    try:
        print fname
        check_midi(fname,THRESHOLD = 0.1)
        return 1
    except Exception as e:
        print e
        return 0
        
def DIR_single2mroll(DIR,cap = 50, para = 12):
    FILES0 = list(os.walk(DIR))[0][-1]
    FILES0 = [os.path.join(DIR,f) for f in FILES0]
#     for f in FILEs:
#         os.remove(f)
    for f in FILES0:
        if f.endswith('.mroll.npy'):
            os.remove(f)
#         assert isinstance(f,str),f
#         fname = f+'.mroll.npy'
#         if os.path.isfile(fname):
#             os.remove(fname)
    #### scan all midi
    FILES = [f.rsplit('.',1)[0] for f in FILES0 if f.endswith(".single_mid")]
    #### recompute mroll
#     FILES = [os.path.join(DIR,f.rstrip('.mroll.npy')+'.mid') for f in FILES0 if f.endswith(".mroll.npy")]
#     FILES = [os.path.join(DIR,f.rstrip('.mroll.npy')) for f in FILES0 if f.endswith(".mroll.npy")]

    FILES = [f+'.single_mid' for f in FILES]
    FILES = FILES[:cap]
#     print FILES[0]
#     break
    suc = mp_map( func_DIR_single2mroll,FILES,para)
#     suc = mp_map(func,FILES,1)    
    print "All:",len(suc)
    print "skipped:",sum(1 for x in suc if x is None)
    print "success:",sum(x for x in suc if x is not None)
    print "Done"
# spwave.read(fname)

# import util_midi
# suc = map(midi_feed,FILES)

def func_DIR_mroll2both(fname,offset = 2,**kwargs):
    try:
        util_midi.mroll_feed(fname,norm = 0, 
                             offset = offset,
                              DEBUG = 1, **kwargs
#                             DEBUG = 1
                            )
#         print fname
        return 1
    except Exception as e:
        print '%s \n %s'%(e,fname)
# func = functools.partial(midi_feed,norm = 1)
# func = functools.partial(midi_feed,norm = 0)
def DIR_mroll2both(DIR,cap = 50,para = 12,**kwargs):
    FILES0 = list(os.walk(DIR))[0][-1]
    FILES0 = [os.path.join(DIR,f) for f in FILES0]
    for f in FILES0:
        if f.endswith('.both.npy'):
            os.remove(f)
        
    FILES = [f for f in FILES0 if f.endswith('.mroll.npy')]
    FILES = FILES[:cap]
    func = functools.partial(func_DIR_mroll2both,**kwargs)
    suc = mp_map(func,FILES, para)
#     suc = mp_map(func,FILES,para)    
    print "All:",len(suc)
    print "skipped:",sum(1 for x in suc if x is None)
    print "success:",sum(x for x in suc if x is not None)
    print "Done"
if __name__=='__main__':

    DIRs = [
#     'sample/MIDI/',
        'sample/MIDI/midiworld/',
#         'sample/MIDI/jsbach/'
       ]
    for DIR in DIRs:
        DIR_mid2single(DIR,cap = 500, para = 8)
    for DIR in DIRs:
        DIR_single2mroll(DIR,cap = 100, para = 7)
    for DIR in DIRs:
        DIR_mroll2both(DIR,cap = 500, para = 7)
    
    print 'TESTS finished!!'