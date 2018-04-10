import numpy as np
import os,subprocess
import matplotlib.pyplot as plt
#%matplotlib inline
import IPython.display as ipd
import keras

#import util_midi
mae = lambda x:np.mean(abs(x))
def abs_logR(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), None)
    y_pred = K.clip(y_pred, K.epsilon(), None)
    return K.sum( K.abs(K.log(y_true / y_pred)), axis=-1)
#    return K.sum( K.abs(y_true- y_pred) * K.abs(K.log(y_true / y_pred)), axis=-1)
#    return K.sum(y_true * K.abs(K.log(y_true / y_pred)), axis=-1)
#def abs_KL_div(y_true, y_pred):
#    y_true = K.clip(y_true, K.epsilon(), None)
#    y_pred = K.clip(y_pred, K.epsilon(), None)
##    return K.sum( K.abs(K.log(y_true / y_pred)), axis=-1)
#    return K.sum( K.abs(y_true- y_pred) * K.abs(K.log(y_true / y_pred)), axis=-1)

def abs_KL_div(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), None)
    y_pred = K.clip(y_pred, K.epsilon(), None)
#    return K.sum( K.abs( (y_true- y_pred) * (K.log(y_true / y_pred))), axis=-1)
    return K.sum( (y_true- y_pred) * (K.log(y_true / y_pred)), axis=-1)
#    return K.sum(y_true * K.abs(K.log(y_true / y_pred)), axis=-1)
cObj = {'abs_KL_div':abs_KL_div,
       'abs_logR':abs_logR,
       'keras':keras}

def load_data(DIR = 'sample/MIDI/'):    
    # fname = 'sample/MIDI/composer-bach-edition-bg-genre-cant-work-0002-format-midi1-multi-zip-number-01.mid'
    FILES = list(os.walk(DIR))[0][-1]
    FILES = [os.path.join(DIR,f) for f in FILES if f.endswith('.both.npy')]
    # for fname in FILES:
    #     print ["timidity -Ow", fname]
    #     out = subprocess.check_call(["timidity","-Ow", fname])
#     Xs = []
#     Ys = []
    data = []
    for datafile in FILES:
        bname = datafile.rstrip('.both.npy')
#        bname = fname.rsplit('.',1)[0]
#        datafile  = '%s.npy' % bname
        if os.path.isfile(datafile):
            d = np.load(datafile)
#             data.append(d)
#             print type(d)
            
            sound, mroll = np.load(datafile)
            X = np.squeeze(sound)
            Y = np.squeeze(mroll)
#             d.pop
            data.append({"name":bname,
                        "X":X,
                        "Y":Y})
#             Xs.append(np.squeeze(sound))
#             Ys.append(mroll)
    return data            
import itertools
def make_buffer(data,truncate = 0,model = None,fs = None):
    Xs = []
    Ys = []
    for d in data:
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
#    Xs = [d.get("X") for d in data]
#    Ys = [d.get("Y") for d in data]
#    for X,Y in itertools.izip(Xs,Ys):
        if model is not None:
            if len(X) > 0:
                pY = model.predict_on_batch(X)
                Y = np.clip(Y - pY,0 ,None)
        Xs += [X]
        Ys += [Y]
    Xs = np.concatenate(Xs,axis = 0)
    Ys = np.concatenate(Ys,axis = 0)
    return Xs,Ys


import matplotlib.pyplot as plt
def plot_model_loss(history):

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential,load_model
import keras
#
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
# from keras.layers.convolutional import Convolution2D

from keras.layers import *
from keras.layers.convolutional import *
class PGAgent:
    def __init__(self, state_size, action_size,model = None,AgentName = 'tst'):
#         self.AgentName = 'pong_minimal-s5L1b-tst'
        self.name = AgentName
#         self.AgentFile = 'Models/%s.h5'%AgentName;
        self.AgentFile="Models/{:}-best_only.hdf5".format(self.name)
        self.LogName = 'Models/%s.log'%AgentName;
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        if model is None:
            model = self._build_model()
        self.model = model
            
        self.summary = self.model.summary;
    def callback_checkpoint(self,save_best_only = False):
        if save_best_only:
            filepath="Models/{name:}-{{epoch:02d}}-{{val_acc:.2f}}.hdf5".format(name = self.name)
        else:
#             filepath="Models/{:}-best_only.hdf5".format(self.name)
            filepath = self.AgentFile
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only = save_best_only , mode='max')
        return checkpoint

# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]


    def _build_model(self):
        model = Sequential()
#         model = Sequential_wrapper()
#         model.add(Reshape((80, 80, 1), input_shape=(self.state_size,)))
        model.add(Reshape((self.state_size,1), input_shape=(self.state_size,)))
#         model.add(Reshape(self.state_size, input_shape=(self.state_size,)))
#         model.add(BatchNormalization(axis = 1))
        model.add(Conv1D(40, (5,), 
#                          subsample=(3,),
#                          border_mode='same',
                         border_mode='valid',
                                activation='relu', init='he_uniform'))
        model.add(Conv1D(30, (5, ), strides=3,
#                          subsample=(1, 1),
#                          border_mode='same',
                         border_mode='valid',
                                activation='relu', init='he_uniform'))
        model.add(Conv1D(20, (5, ), strides=3,
#                          subsample=(1, 1), 
#                          border_mode='same',
                         border_mode='valid',
                                activation='relu', init='he_uniform'))
        model.add(Conv1D(20, (5, ), strides=3,
#                          subsample=(1, 1),
#                          border_mode='same',
                         border_mode='valid',
                                activation='relu', init='he_uniform'))
        model.add(Conv1D(20, (5, ), strides=3,
#                          subsample=(1, 1),
#                          border_mode='same',
                         border_mode='valid',
                                activation='relu', init='he_uniform'))
#         model.add(Convolution2D(25, (6, 6), subsample=(3, 3), border_mode='same',
#                                 activation='relu', init='he_uniform'))
#         model.add(Convolution2D(5, (6, 6), subsample=(1, 1), border_mode='same',
#                                 activation='relu', init='he_uniform'))
#         model.add(Convolution2D(5, (6, 6), subsample=(1, 1), border_mode='same',
#                                 activation='relu', init='he_uniform'))
#         model.add(Convolution2D(5, (6, 6), subsample=(1, 1), border_mode='same',
#                                 activation='relu', init='he_uniform'))
        model.add(Flatten())
#         model.add(Dense(20, activation='relu', init='he_uniform'))
#         model.add(Dense(20, activation='relu', init='he_uniform'))
        model.add( Dense(self.action_size, activation='softmax'))
    
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt
                     ,metrics=['accuracy'])
        return model

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def act(self, state):
        # state = state.reshape([1, state.shape[0]])
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self,rewards,batch_size = 2000,verbose = 0):
        gradients = np.vstack(self.gradients)
        # rewards = np.vstack(self.rewards)
        # rewards = self.discount_rewards(rewards)
        rewards = (rewards - np.mean(rewards,keepdims=1)) / np.std(rewards)
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
#         print(X.shape)
#         self.model.train_on_batch(X, Y)
#         X = np.expand_dims(X,axis = 1)
#         Y = np.expand_dims(Y,axis = 1)
    
#         gen = ((X[i],Y[i]) for i in range(len(X)));
#         zipped = zip(X,Y)
        Xgen = chunks(X,batch_size);
        Ygen = chunks(Y,batch_size);
        gen = (x for x in zip(Xgen,Ygen))
#         gen1 = ((x,y) for x,y in zip(chunks(X,batch_size).next(),chunks(Y,batch_size).next()))
#         print(gen1.next()[1].shape)
#         print(chunks(X,batch_size).next().shape)
#         print()
        bmax = max(X.shape[0]//batch_size,1)
#         print(gen)
#         print(bmax)
        try:
            self.model.fit_generator(gen, steps_per_epoch = bmax, epochs = 1,verbose = verbose, max_q_size=1)
        except StopIteration:
            pass
#         self.model.fit(X,Y,  epochs = 1,verbose = verbose)
    
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def load(self, fname=None):
        if not fname:
            fname = self.AgentFile;
        global episode
        self.model = load_model(fname, custom_objects=cObj);
        self.summary = self.model.summary;
        
    def readlog(self, LogName=None):
        if not LogName:
            LogName = self.LogName;
        with open(LogName,'rb') as f:
                first = f.readline()      # Read the first line.
                if not first.rstrip('\n'):
#                     print('nothing!')
                    self.episode = 0;
                else:
                    f.seek(-2, 2)             # Jump to the second last byte.
                    while f.read(1) != b"\n": # Until EOL is found...
                        f.seek(-2, 1)         # ...jump back the read byte plus one more.
                    last = f.readline() 
                    lst = last.split('\t');
                    eind = lst.index('Episode')+1;
                    self.episode = int(lst[eind]);

    def save(self, fname=None):
        if not fname:
            fname = self.AgentFile;
        self.model.save(fname);
    def writelog(self, msg, LogName = None):
        if not LogName:
            LogName = self.LogName
        with open(LogName,'a+') as LogFile:
            LogFile.write(msg+'\n');
    def newlog(self):
        open(self.LogName,'w').close();
        
def flatten_param(opt_par):
    flat_par = '-'.join(['_'.join(str(y) for y in x) for x in opt_par.items()]).replace('.','Dot')
    return flat_par




import copy
import matplotlib 
#import util_midi
def log_plot(Z1,eps = 1E-5):
    Z1 = abs(Z1)
    Z1 += eps
    plt.pcolormesh(Z1,
                   alpha = 0.5,
                   norm=matplotlib.colors.LogNorm(vmin=Z1.min(), vmax=Z1.max())
                  )
def compare(log = 1, mode = 'test',DATA = None,best_agent = None,chroma = False,overlap = 1, loss_func = None,callbacks = None):
    FSIZE = [12,4]
    if best_agent is None:
        best_agent = copy.copy(agent)
        best_agent.load()
    plt.figure(figsize = FSIZE)

    if DATA is None:
        if mode == 'test':
            pXs = data[0]["X"][:100]
            pYs_exp = data[0]["Y"][:100]
        else:
            pXs = Xs[:100]
            pYs_exp = Ys[:100]
    else:
        if isinstance(DATA,dict):
            pXs = DATA.get('X')
            pYs_exp = DATA.get('Y')
        else:
            pXs,pYs_exp = DATA
#     assert not (pXs[0] - Xs[0]).any()
#     pXs = Xs[:500]
#     pYs_exp = Ys[:500]
    pYs_act = best_agent.model.predict_on_batch(pXs) 
    pYs_act += 1E-5
#     compare(log = 1)
    if chroma:
        if pYs_exp.shape[-1]==128:
            pYs_exp = mroll2chroma(pYs_exp,norm = 1)
        if pYs_act.shape[-1]==128:
            pYs_act = mroll2chroma(pYs_act,norm = 1)
    print pXs.shape, pYs_exp.shape
    plt.pcolormesh(pYs_exp.T,alpha = 0.75)
    if overlap:
        pass
    else:
        plt.figure(figsize = FSIZE)
        
    Z1 = pYs_act.T    
    if log:
        log_plot(Z1)
    else:
        plt.pcolormesh(Z1)
#     if overlap:
#         pass
#     else:
    if 1:
        plt.figure(figsize = FSIZE)
    if loss_func is not None:
        loss = loss_func(pYs_exp, pYs_act)
    else:
        loss = (pYs_act - pYs_exp)
    
    Z1 = loss.T
    if log:
        log_plot(Z1)
    else:
        plt.pcolormesh(Z1)        
#     plt.pcolormesh( loss.T )
        
    if chroma:
        YLIM = [0,12]
    else:
        YLIM = [40,90]
    plt.gca().set_ylim(YLIM)
#     plt.gca().set_xlim(0,200)
#     plt.gca().set_ylim(YLIM)
#     plt.figure()
#     pYs_act = np.hstack([pYs_act,pYs_act])
    print pYs_act.shape
    print pYs_exp.shape
#     pYs = np.vstack
    ipd.display(ipd.Audio(util_midi.midi_roll_play(pYs_act.T),rate = 16000))
    ipd.display(ipd.Audio(util_midi.midi_roll_play(pYs_exp.T),rate = 16000))
    cpXs = np.concatenate(pXs,axis = 0)
    ipd.display(ipd.Audio(cpXs,rate = 16000))
    


import multiprocessing as mp
def mp_map(f,lst,n_cpu,**kwargs):
    if n_cpu >1:
        p = mp.Pool(n_cpu)
        OUTPUT=p.map(f,lst)
        p.close()
    else:
        OUTPUT = map(f,lst)
    return OUTPUT



###### Gradient of a model
def make_gradient_function(model):
    input_tensors = [model.inputs[0], # input data
                     model.sample_weights[0], # how much to weight each sample by
                     model.targets[0], # labels
                     K.learning_phase(), # train or test mode
    ]
    weights= model.trainable_weights
    gradients = model.optimizer.get_gradients(model.total_loss, weights)
    gradF= K.function(inputs=input_tensors, outputs=gradients)
    model.gradF = gradF
    return gradF
# make_gradient_function(m)
def get_gradients(inputs, outputs=None, weight = None, gradF = None):
    '''
    Get a gradient using a model
    '''
    if isinstance( gradF,keras.models.Model):
        if hasattr(gradF,'gradF'):
            gradF = gradF.gradF
        else:
            gradF = make_gradient_function( gradF)
    if isinstance(inputs,dict):
        ddict = inputs
        inputs = ddict['input']
        outputs  = ddict['output']
    outputs = np.array(outputs)
    if outputs.ndim==1:
        outputs=  outputs [:,None]
    if weight is None:
        weight = [1] * len(inputs)
    return gradF([inputs,weight,outputs])   

from util import *
class GradientCallback(keras.callbacks.Callback):
    def __init__(self,test_data,lossfunc = mae):
        self.test_data = test_data
        self.record = {'ep_end':[]}
        self.lossfunc = lossfunc
#         self.test_data = test_data
        pass

    def on_epoch_end(self, epoch, logs={'params'}):
#         self.model.gradF()
        gd = get_gradients( self.test_data,gradF = self.model.gradF)
        gdnorm = self.lossfunc(0,flatten(gd))
        self.record.get('ep_end').append(gdnorm)
        
def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
            
            
def train_until_converge(m,tdata, maxEPC = 5, tol=0.05,**kwargs):
    if not hasattr(m,'gradF'):
        m.gradF = make_gradient_function(m)
    gd_hist = GradientCallback(tdata,lossfunc = mae)
    X = tdata['input']
    y = tdata['output']
    for i in range(maxEPC):
        general_hist = m.fit(X,y,batch_size=len(X),epochs=1,callbacks=[gd_hist],**kwargs)
        last_gd = gd_hist.record['ep_end'][-1]
#         print last_gd
        if last_gd < tol:
            break
        if i+1 == maxEPC:
            print '[WARN]:gradient not converged: %.3f > %.3f' %(last_gd,tol)
    return m
