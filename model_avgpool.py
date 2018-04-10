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
def make_model(input_size = 800, output_size = 32,self = None,Nl = 6):
    Ns = 2 * np.clip(np.arange(1, Nl+1),None,15)
    Ss = [3,] * Nl
    Ns = np.linspace(1,np.log2(output_size),Nl)
    Ns = (2**Ns).astype(int)
    print Ns
    Ws = np.multiply(1,Ss)
    import keras.backend as K

    main_input = Input(shape=(800,), 
#                        dtype='int32', name='main_input'
                      )
    curr = Reshape((input_size,1,1), input_shape=(input_size,))(main_input)

    imax = len(Ns)
    for i in range(imax):
        N = Ns[i];S = Ss[i];W = Ws[i]
#         if i < 6:
        if 1:
            spec = {'dilation_rate': (S,1)}
        else:
            spec = {'strides': (S,1)}
        spec.update(common)
        curr = Conv2D( N, ( W, 1),
                                     border_mode='valid',
#                              border_mode='dilated',
                                    activation=ACT_FUNC, init='he_uniform',
                     **spec)(curr)
        if i+1==imax:
#             curr = Dense(12)(curr)
            pass
    out1 = AvgPool2D((curr._keras_shape[1],1) )(curr)
#     total = keras.layers.Lambda(lambda x: K.clip(x,0.01,None))(
#             Dense(1,activation='relu',**common)(curr))
#     total = Flatten()(AvgPool2D((total._keras_shape[1],1) )(total))
    curr = Flatten()(curr)
    concat = curr
#     print K.eval(K.shape(concat))
    print concat._keras_shape

#     print concat.shape
#     out1 = Dense(48,activation='softmax',**common)(curr)
    out2 = None
    out3 = None
#     total= keras.layers.Lambda(lambda x: K.clip(x,0.01,None))(
#         Dense(1,activation='relu',**common)(concat))
    Llst = [Flatten()(out) for out in [out1,out2,out3] if out is not None]
#     Llst = [BatchNormalization(axis=1)(out) for out in [out1,out2,out3] if out is not None]
    if len(Llst) > 1: 
        main_output = keras.layers.add(Llst)
    else:
        main_output = Llst[0]
#     total = keras.layers.Lambda(lambda x: K.clip(x,0.01,None))(
#             Dense(1,activation='relu',**common)(main_output))

    ###### Optionally predicting the partition as well
#     main_output = keras.layers.Dense(output_size,activation='linear')(main_output)
#     main_output = keras.layers.Activation('softmax')(main_output)
#     main_output = keras.layers.multiply([main_output,total])
    
    opt = Adam()
#     model.compile(loss=LOSS, optimizer=opt
#                  ,metrics=['accuracy'])
    model = Model(inputs=[main_input,
#                           noise_input
                         ], 
                  outputs=[main_output])
    return model