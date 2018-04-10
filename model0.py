import numpy as np
import keras.backend as K
from util_nn import *
def reg_l2_avg(const=0.1):
    def l2_reg(weight_matrix):
        return const * K.mean(K.square(weight_matrix))
    return l2_reg
def abs_KL_div(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), None)
    y_pred = K.clip(y_pred, K.epsilon(), None)
#    return K.sum( K.abs( (y_true- y_pred) * (K.log(y_true / y_pred))), axis=-1)
    return K.sum( (y_true- y_pred) * (K.log(y_true / y_pred)), axis=-1)

REG = reg_l2_avg(0.1)
common = {'kernel_regularizer':REG,
         'bias_regularizer':REG}
alias = 'merge'
ACT_FUNC = "relu"
# # ACT_FUNC = "selu"
# tf.device('/gpu:0')
# LOSS = 'sparse_categorical_crossentropy'
LOSS = 'binary_crossentropy'
from keras.models import Model,Sequential

import keras
import keras.backend

# Ns = [2,3,4,5,6][::-1]
Nl = 17
# Ns = [30]*Nl
# Ns = 8 * np.clip(np.arange(1, Nl+1),None,15)
Ns = 2 * np.clip(np.arange(1, Nl+1),None,15)
# Ss = [2,4,8,16,16,16,16,16,100,100,100]
Ss = (1.25 * np.clip(np.arange(1, Nl+1),None,5)).astype('int')
# Ss = [7] * Nl
# Ss = [2,4,8,16,32,64,128,256]
Ws = 2 * np.clip(np.arange(1, Nl+1),None,5)
# LOSS = 'binary_crossentropy'
LOSS = abs_KL_div
# LOSS = 'kullback_leibler_divergence'
# common = 
def make_model(input_size = 800, output_size = 32,self = None):
    import keras.backend as K
    model = Sequential()
#         model = Sequential_wrapper()
#         model.add(Reshape((80, 80, 1), input_shape=(self.state_size,)))
    
#     model.add(Reshape((self.state_size,1), input_shape=(self.state_size,)))
    model.add(Reshape((input_size,1,1), input_shape=(input_size,)))

#     S = 85;niter = 1
    S = 9;niter = 2
#     S = 4;niter = 3    
    niter = 4
    n = S
    nextlayer = lambda x:(    x//n - 1)
    
    N0 = 64
    S2 = 30
    N2 = 10
    Lx = 800
    
#         if i == 0:

#     #         N = 1 if i==0 else N0
#             N = N0
#             if i==0:
#                 N2_curr = 1
#             else:
#                 N2_curr = N2
    N2_curr = 1
    N = 10
    # Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
    # Note that we can name any layer by passing it a "name" argument.
    main_input = Input(shape=(800,), 
#                        dtype='int32', name='main_input'
                      )
    curr = Reshape((input_size,1,1), input_shape=(input_size,))(main_input)

    imax = len(Ns)
    for i in range(imax):
        N = Ns[i];S = Ss[i];W = Ws[i]

        curr = Conv2D( N, ( W, 1),
#         curr = Conv2D( N, ( S*8, int(curr._keras_shape[2])),
#         curr = Conv2D( N, ( S*8, curr._keras_shape[2]//2),
                  dilation_rate = (S,1),
#                    strides=(S, 1),
                                     border_mode='valid',
#                              border_mode='dilated',
                                    activation=ACT_FUNC, init='he_uniform',
                     **common)(curr)
        if i+1==imax:
#             curr = Dense(12)(curr)
            pass
    curr = Flatten()(curr)
#     print curr.shape[1]
#     print curr._keras._shape[0]
#     shortcut = keras.layers.Conv2D(1, kernel_size=(1, 1), strides=_strides, padding='same')(curr)

    concat = curr
#     main_output = Dense(128,activation='softmax')(Flatten()(concat))
#     main_output = Dense(12,activation='softmax')(Flatten()(concat))
#     main_output = Dense(128,activation='sigmoid')(Flatten()(concat))

#     curr = Dense(128,activation='relu')(curr)
#     total= Dense(1,activation='relu')(concat)
#     curr = keras.layers.add([curr , total])
#     main_output = Activation(activation = 'softmax')(curr)

#     out1 = Dense(1,activation='sigmoid')(concat)
    out1 = Dense(output_size,activation='softmax',**common)(concat)
#     out1 = Dense(128,activation='softmax')(concat)
#     out2 = Dense(128,activation='softmax')(curr)
    out2 = None
#     out3 = Dense(128,activation='softmax')(curr)
    out3 = None
    total= keras.layers.Lambda(lambda x: K.clip(x,0.01,None))(
        Dense(1,activation='relu',**common)(concat))
#     total =         Dense(1,activation='relu')(concat)
#     total= keras.layers.Lambda(lambda x: keras.backend.clip(x,0.01,None))(
#         Dense(1,activation='relu')(main_input))
#     total= keras.layers.Lambda(lambda x: keras.backend.clip(x,0.01,1.01))(
#         Dense(1,activation='relu')(main_input))
# BatchNormalization    
    Llst = [(out) for out in [out1,out2,out3] if out is not None]
#     Llst = [BatchNormalization(axis=1)(out) for out in [out1,out2,out3] if out is not None]
    if len(Llst) > 1: 
        main_output = keras.layers.add(Llst)
    else:
        main_output = Llst[0]
    main_output = keras.layers.multiply([main_output,total])
#     main_output = total
    
#     main_output = keras.layers.multiply([main_output,total])

    #     shortcut = keras.layers.Lambda(lambda x: x + 0, )(curr)
#     noise_input = keras.models.Input((1,))
    
#     curr = keras.layers.GaussianNoise(.1)(curr)
#     curr = concatenate([curr,
# #                         noise_input,
#                        ]
#                         )
#                       keras.layers.GaussianNoise(.1)()])
#     curr = keras.layers.add(
#         [
#             keras.layers.GaussianNoise(.1),
# #             Dense(128,activation = 'relu')(shortcut),
#             Dense(128,activation = 'relu')(curr),
# #             shortcut
#         ]
#     )
#     main_output = Dense(128,activation='softmax')(curr)
#     main_output = Dense(12,activation='softmax')(Flatten()(concat))
    opt = Adam()
    model.compile(loss=LOSS, optimizer=opt
                 ,metrics=['accuracy'])
    model = Model(inputs=[main_input,
#                           noise_input
                         ], 
                  outputs=[main_output])
    return model
m = make_model()
m.summary()