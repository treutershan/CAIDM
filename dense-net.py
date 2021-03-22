import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jarvis.train import datasets
from jarvis.train.client import Client
from jarvis.utils.general import tools as jtools
from jarvis.utils.display import imshow
from tensorflow.keras import Input, Model, models, layers, metrics, callbacks, regularizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import losses, optimizers
from jarvis.utils.general import gpus, overload
gpus.autoselect()

# --- Preprocess data with overload
@overload(Client)
def preprocess(self, arrays, **kwargs):
    arrays['ys']['lbl'] *= 2.5
    arrays['ys']['lbl'] = arrays['ys']['lbl'].clip(max=1.0)
    return arrays

# --- Create client and generators
paths = jtools.get_paths('xr/breast-fgt')
client = Client('{}/data/ymls/client.yml'.format(paths['code']))
gen_train, gen_valid = client.create_generators()

# --- Create model inputs
inputs = client.get_inputs(Input)

# --- Define lambda functions
conv3 = lambda x, filters : layers.Conv3D(
    kernel_size=(1, 3, 3),
    filters=filters,
    strides=1,
    padding='same',
    kernel_regularizer=regularizers.l2(0.01))(x)

conv1 = lambda x, filters : layers.Conv3D(
    kernel_size=(1, 1, 1),
    filters=filters,
    strides=1,
    padding='same')(x)

pool = lambda x : layers.AveragePooling3D(
    pool_size=(1, 2, 2),
    strides=(1, 2, 2),
    padding='same')(x)

norm = lambda x : layers.BatchNormalization()(x)
relu = lambda x : layers.LeakyReLU()(x)
concat = lambda a, b : layers.Concatenate()([a, b])
#drop = lambda x : layers.Dropout(0.2)(x)

dense = lambda x, k : conv3(relu(norm(x)), filters=k)

bneck = lambda x, b : conv1(relu(norm(x)), filters=b)

def dense_block(x, k=8, n=3, b=1):
    ds_layer = None
    for i in range(n):
        cc_layer = concat(cc_layer, ds_layer) if ds_layer is not None else x
        #bn_layer = bneck(cc_layer, b * k) if i >= b else cc_layer
        #ds_layer = dense(bn_layer, k)
        ds_layer = dense(cc_layer, k)
    return concat(cc_layer, ds_layer)

# --- Create DenseNet
k = 8
#n = 3
b = 1
dense_block_ = lambda x, n : dense_block(x, k, n, b)
trans = lambda x, b : pool(bneck(x, b))

b0 = conv3(inputs['dat'], filters=8)
b1 = pool(dense_block_(b0, 3)) # n = 6, 24
b2 = pool(dense_block_(b1, 3)) # n = 12, 32

k = 12
b = 2
dense_block_ = lambda x, n : dense_block(x, k, n, b)

b3 = trans(dense_block_(b2, 3), 40) # n = 24, 40
b4 = trans(dense_block_(b3, 3), 48)
b5 = trans(dense_block_(b4, 6), 56) # n = 16, 128
bf = dense_block_(b5, 4)

# --- Flatten via pooling
f0 = layers.AveragePooling3D(pool_size=(1, bf.shape[2], bf.shape[3]), padding='same')(bf)
#f0 = layers.GlobalAveragePooling3D()(bf)

# --- Flatten via "reshape"
#f0 = layers.Reshape((1, 1, 1, bf.shape[2] * bf.shape[3] * bf.shape[4]))(bf)

# --- Create logits
logits = {}
#logits['lbl'] = layers.AveragePooling3D(pool_size=(1, bf.shape[2], bf.shape[3]), padding='same', name='lbl')(bf)
logits['lbl'] = layers.Conv3D(filters=1, kernel_size=(1, 1, 1), activation='sigmoid', name='lbl')(f0)

# --- Create model
model = Model(inputs=inputs, outputs=logits)

# --- Compile model
model.compile(
    optimizer=optimizers.Adam(learning_rate=5e-5),
    #loss=losses.Huber(delta=0.042),
    loss=losses.MeanAbsoluteError(),
    metrics=['mse', 'mae', 'mape'],
    experimental_run_tf_function=False)

# --- Load data into memory for faster training
client.load_data_in_memory()

# --- TensorBoard
#tensor_board = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

# --- Learning rate scheduler
lr_scheduler = callbacks.LearningRateScheduler(lambda epoch, lr : lr * 0.996)

# --- csv Callback
num = '0091'
path = '/home/treuters/breast-density/dense-net/experiments/exp-'+ num + '/'
csv_logger = callbacks.CSVLogger(path + 'history-' + num + '.csv', append=True)

# --- Checkpoint callback
file = path + 'model-' + num + '-{mae:.3f}-{epoch:03d}.hdf5'
checkpoint = callbacks.ModelCheckpoint(file, verbose=1, save_best_only=False, \
                                       mode='auto', period=5)

# --- Train model
model.fit(
    x=gen_train,
    steps_per_epoch=100,
    epochs=2000,
    validation_data=gen_valid,
    validation_steps=50,
    validation_freq=1,
    callbacks=[lr_scheduler, csv_logger, checkpoint])
