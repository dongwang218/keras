
import os
import itertools
import re
import time
import datetime
import random
import editdistance
import numpy as np
from scipy import ndimage
from keras import backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Layer, Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Reshape, Lambda, merge, Permute, TimeDistributed
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
import math, glob, sys, os, re
from PIL import Image, ImageChops
import PIL.ImageOps

np.random.seed(55)

image_height = 128
image_width = 1024 # int(sys.argv[1]) # 1024
trimBg = Image.new('L', (3000,1000), 255)
inputAvg = 255/2.
inputStd = 255/2.

def read_image(imagePath):
  # Load all stroke points for the current line as inputs
  image = Image.open(imagePath)
  image = image.crop(ImageChops.difference(image, trimBg).getbbox())
  imageWidth = int(image.size[0] / float(image.size[1]) * image_height)
  image = image.resize((image_width, image_height))
  # image = cv2.threshold(np.array(image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  inputs = (np.array(image) - inputAvg) / inputStd
  return inputs

def get_ordered_chars():
  labels = {}
  print "Loading labels"
  for fileName in glob.glob('ascii/*/*/*.txt'):
    asciiFile = open(fileName, 'r')
    text = re.sub(r'.*[\r\n]+CSR:\s*[\r\n]+', '', asciiFile.read(), 1, re.DOTALL)
    for char in text:
      if char == '\n' or char == '\r': continue
      if char not in labels:
        labels[char] = len(labels)
      asciiFile.close()
  chars = sorted(labels.items(), key = lambda x: x[1])
  return [x[0] for x in chars] + [' ']


def decode_batch(out, ordered_chars):
  out_best = list(np.argmax(out, 1))
  out_best = [k for k, g in itertools.groupby(out_best)]
  # 26 is space, 27 is CTC blank char
  outstr = ''
  for c in out_best:
    if c < len(ordered_chars)-1:
      outstr += ordered_chars[c]
  return outstr

K.set_learning_phase(0)

# Network parameters
conv_num_filters = 16
filter_size = 3
pool_size_1 = 4
pool_size_2 = 2
time_dense_size = 32
rnn_size = 512
time_steps = image_width / (pool_size_1 * pool_size_2)

weights_file = sys.argv[1]
image_file = sys.argv[2]
truth = sys.argv[3]

ordered_chars = get_ordered_chars()

act = 'relu'
input_shape = (image_height, image_width, 1)
with K.tf.Session() as sess:
  input_data = Input(name='the_input', shape=input_shape, dtype='float32')
  inner = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                        activation=act, name='conv1')(input_data)
  inner = MaxPooling2D(pool_size=(pool_size_1, pool_size_1), name='max1')(inner)
  inner = BatchNormalization()(inner)
  inner = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                        activation=act, name='conv2')(inner)
  inner = MaxPooling2D(pool_size=(pool_size_2, pool_size_2), name='max2')(inner)
  inner = BatchNormalization()(inner)
  inner = Permute(dims=(2, 1, 3), name='permute')(inner)

  conv_to_rnn_dims = (image_width / (pool_size_1 * pool_size_2), (image_height / (pool_size_1 * pool_size_2)) * conv_num_filters)
  inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

  # cuts down input size going into RNN:
  inner = TimeDistributed(Dense(time_dense_size, activation=act, name='dense1'))(inner)

  inner = Dropout(0.25)(inner)

  # Two layers of bidirecitonal GRUs
  # GRU seems to work as well, if not better than LSTM:
  gru_1 = GRU(rnn_size, return_sequences=True, name='gru1', dropout_W = 0.1, dropout_U = 0.1)(inner)
  gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, name='gru1_b', dropout_W = 0.1, dropout_U = 0.1)(inner)
  gru1_merged = merge([gru_1, gru_1b], mode='sum')
  gru_2 = GRU(rnn_size, return_sequences=True, name='gru2')(gru1_merged)
  gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True)(gru1_merged)

  # transforms RNN output to character activations:
  inner = TimeDistributed(Dense(85, name='dense2'))(merge([gru_2, gru_2b], mode='concat'))
  y_pred = Activation('softmax', name='softmax')(inner)
  model = Model(input=[input_data], output=y_pred)
  #model.summary()
  model.load_weights(weights_file)

  inputs = read_image(image_file).astype(np.float32).reshape(1, image_height, image_width, 1)
#  outputs = model(inputs)
#  print outputs

  probability = sess.run(model.output, {input_data: inputs})[0]
  #print probability
  sentence = decode_batch(probability, ordered_chars)
  edit_dist = editdistance.eval(sentence, truth)
  print truth
  print sentence
  print edit_dist

#K.tf.ConfigProto
