from util_nn import *
from evaluator import *
from pymisca.util import *
# np.random.seed(0)
# agent,r = pretrain()
N = 300000


from pymisca.UGD import UGD
def subsample(DATA,batch_size):
    Xs,Ys = DATA
    idx = np.random.randint(0,len(Xs),batch_size,)
    subXs = Xs[idx]
    subYs = Ys[idx]
    return subXs,subYs
def make_gen(DATA,batchsize):
    def gen():
        while True:
            idx = np.random.randint(0,len(Xs),batchsize,)
#             idx = np.random.choice(len(Xs),batchsize,replace=0)
            subXs = Xs[idx]
            subYs = Ys[idx]
            yield subXs,subYs
#             yield subsample(DATA,batchsize)
    return gen

import threading
class createBatchGenerator:
    def __init__(self, DATA, batch_size=32):
        self.DATA = DATA
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.next = self.__next__

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            Xs,Ys = self.DATA
            idx = np.random.randint(0,len(Xs),self.batch_size,)
#             idx = np.random.choice(len(Xs),batch_size,replace=1)
            subXs = Xs[idx]
            subYs = Ys[idx]
            return subXs,subYs
#     def next(self):
#         self.next()


def helper_make_buffer(d,truncate=0,fs = None,model = None ):
    X = d.get("X")
    Y = d.get("Y")
    if truncate > 0:
        X = X[:truncate + 1]
        Y = Y[:truncate + 1]
    elif truncate < 0:
        X = X[truncate:]
        Y = Y[truncate:]
    if fs is not None:
        #X = X[:,fs]
        Y = Y[:,fs]
    if model is not None:
        if len(X) > 0:
            pY = model.predict_on_batch(X)
            Y = np.clip(Y - pY,0 ,None)
    out  = X,Y  
#     out = map(np.ndarray.tolist,out)
    return out
def make_buffer(data,truncate = 0,para=1,fs=None,**kwargs):
    helper = functools.partial(helper_make_buffer, truncate=truncate,
                              fs = fs)
    out = mp_map(helper, data, para)
    Xs,Ys = zip(*out)
    print np.shape(Xs),np.shape(Xs[0])
    print np.shape(Ys),np.shape(Ys[0])
#     Xs = sum(map(np.ndarray.tolist,Xs),())
#     Ys = sum(map(np.ndarray.tolist,Ys),())
    Xs = np.concatenate(Xs)
    Ys = np.concatenate(Ys)
    return Xs,Ys
#     return np.concatenate(Xs),np.concatenate(Ys)
    


def helper_load_data(datafile):
#     for datafile in FILES:
    bname = datafile.rstrip('.both.npy')
    if os.path.isfile(datafile):
        d = np.load(datafile)
        sound, mroll = np.load(datafile)
        X = np.squeeze(sound)
        Y = np.squeeze(mroll)
        d = {"name":bname,
                    "X":X,
                    "Y":Y}
        return d
    return None
def load_data(DIR = 'sample/MIDI/',para=1,nfile=0):    
    # fname = 'sample/MIDI/composer-bach-edition-bg-genre-cant-work-0002-format-midi1-multi-zip-number-01.mid'
    FILES = list(os.walk(DIR))[0][-1]
    FILES = [os.path.join(DIR,f) for f in FILES if f.endswith('.both.npy')]
    if nfile:
#         FILES = np.take(FILES,np.random.choice(FILES,nfile,replace=1))
        FILES = np.random.choice(FILES,nfile,replace=1)
    # for fname in FILES:
    #     print ["timidity -Ow", fname]
    #     out = subprocess.check_call(["timidity","-Ow", fname])
#     Xs = []
#     Ys = []

    out = mp_map(helper_load_data,FILES,para)
    data = [ x for x in out if not x is None]
    return data
def check_readable(DIR):
    FILES0 = list(os.walk(DIR))[0][-1]
    FILES0 = [os.path.join(DIR,f) for f in FILES0]
    for f in FILES0:
        if f.endswith('both.npy'):
            try:
                _ = np.load(f)
            except:
                print f
                os.remove(f)
        else:
            pass

# def main():
# delist = 'DATA','cDATA','tDATA','data','Xs','Ys'
# for var in delist:
#     if var in locals().keys():
#         exec 'del %s' %var
def load_from_dir(DIR='sample/MIDI/jsbach',para=1,nfile=150,
                 span=[48,96]):
    print "LOADING data from :%s"%DIR
# para = 1
# nfile = 150
# if 1:
#     DIRs = [
#     #         'sample/MIDI/',
#     #         'sample/MIDI/midiworld/',
# #             'sample/MIDI/jsbach.aug',
#             'sample/MIDI/jsbach',
# #         'sample/MIDI/artificial/',
# #            ]
#     DATA = []
#     cDATA = []
#     tDATA = {}
#     para = 1
#     for DIR in DIRs:
    if 1:
        data = load_data(DIR,para=para,nfile = nfile)
        fs= np.arange(*span)
    #     fs = np.arange(72,72+12)
    #     fs = np.arange(bfreq,bfreq+1)
        Xs,Ys= make_buffer(data[1:],truncate = N,
                           model=None,
                           fs = fs,
                           para=para,
                          )
#         tDATA[DIR] = make_buffer([data[0]],truncate = 150,model=None,fs = fs)
    #     Ys = Ys.sum(axis = 1)
    
    
        SUM = Ys.sum(axis = 1,keepdims =1 )
        idx = np.squeeze(SUM!=0)
        Xs = Xs[idx]
        Ys = Ys[idx]
        cDATA = (Xs,Ys)
    return cDATA
if __name__=='__main__':
    cDATA = load_from_dir('sample/MIDI/jsbach');