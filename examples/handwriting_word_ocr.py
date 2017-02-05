'''
Train iam offline handwriting recognizer. Get val_loss: 18.1516

based on the image_ocr example by Mike Henry
https://github.com/mbhenry/
'''

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
import argparse
import cPickle as pickle

np.random.seed(55)

image_height = 64 #128
output_size = 85
BORDER_SIZE_Y = 0
BORDER_SIZE_X = 16

def speckle(img):
  severity = np.random.uniform(0, 0.6)
  blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
  img_speck = (img + (blur * (img != 1)))
  img_speck[img_speck > 1] = 1
  img_speck[img_speck <= 0] = 0
  return img_speck

def read_image(bool_arr, isTrain, add_noise, image_width):
  """foreground is True, add border to top and bottom, add random speckle if add_noise"""
  raw_height, raw_width = bool_arr.shape
  #assert raw_height == image_height
  new_height = image_height - 2 * BORDER_SIZE_Y
  new_width = min(image_width - 2 * BORDER_SIZE_X, int(raw_width / float(raw_height) * new_height))
  image = Image.fromarray(bool_arr.astype(np.uint8)*255).resize((new_width, new_height))
  if isTrain and add_noise:
    image = image.rotate(np.random.uniform(-3, 3))
    y = 0 if BORDER_SIZE_Y > 0 else int(np.random.uniform(0, 2*BORDER_SIZE_Y))
    x = int(np.random.uniform(0, 2*BORDER_SIZE_X))
  else:
    x = BORDER_SIZE_X
    y = BORDER_SIZE_Y
  standard_image = np.zeros((image_height, image_width), dtype = np.float)
  standard_image[y:(y+new_height), x:(x+new_width)] = np.asarray(image) / 255.0
  if isTrain and add_noise:
    standard_image = speckle(standard_image)
  total_width = new_width + 2*BORDER_SIZE_X
  return (standard_image, total_width)

# Uses generator functions to supply train/test with
# data. Image renderings are text are created on the fly
# each time with random perturbations

class TextImageGenerator(keras.callbacks.Callback):

  def __init__(self, downsample_width, minibatch_size, image_width, add_noise):
    self.add_noise = add_noise
    self.image_width = image_width
    self.max_image_width = 0
    self.labels = {}
    self.trainset = []
    self.valset = []
    self.testset = []
    self.max_line_length = 10
    self.trainset_index = 0
    self.valset_index = 0
    self.testset_index = 0
    self.downsample_width = downsample_width
    self.minibatch_size = minibatch_size

  def get_output_size(self):
    size = len(self.labels) + 1
    assert output_size == size
    return size

  def get_num_trainset(self):
    return len(self.trainset)

  def get_num_valset(self):
    return len(self.valset)

  def get_ordered_chars(self):
    chars = sorted(self.labels.items(), key = lambda x: x[1])
    return [x[0] for x in chars]

  # num_words can be independent of the epoch size due to the use of generators
  # as max_string_len grows, num_words can grow
  def read_iam(self, train_file, val_file, label_file, min_length, max_length):
    self.labels = pickle.load(open(label_file))
    print('total labels', self.get_output_size())
    self.trainset = [x for x in pickle.load(open(train_file)) if len(x[1]) >= min_length and len(x[1]) <= max_length]
    self.valset = [x for x in pickle.load(open(val_file)) if len(x[1]) >= min_length and len(x[1]) <= max_length]
    self.max_line_length = max(max([len(x[1]) for x in self.trainset]), max([len(x[1]) for x in self.valset]))
    print('iam: trainset %d, valset %d, max_line_length %d' % (self.get_num_trainset(), self.get_num_valset(), self.max_line_length))

  # each time an image is requested from train/val/test, a new random
  # painting of the text is performed
  def get_batch(self, size, train):
    assert K.image_dim_ordering() == 'tf'
    X_data = np.ones([size, image_height, self.image_width, 1])

    labels = np.ones([size, self.max_line_length])
    input_length = np.zeros([size, 1])
    label_length = np.zeros([size, 1])
    source_str = []

    while len(source_str) < size:
      i = len(source_str)
      if train == 'train':
        data = self.trainset
        self.trainset_index = (self.trainset_index + 1) % len(self.trainset)
        index = self.trainset_index
      elif train == 'val':
        data = self.valset
        self.valset_index = (self.valset_index + 1) % len(self.valset)
        index = self.valset_index
      else:
        data = self.testset
        self.testset_index = (self.testset_index + 1) % len(self.testset)
        index = self.testset_index

      inputs, width = read_image(data[index][0], train == 'train', self.add_noise, self.image_width)
      label_len = len(data[index][1])
      data_len = int(math.ceil(float(width) / self.downsample_width))
      if data_len < label_len:
        print('Warning: image too short', data_len, label_len, data[index][1])
        continue

      X_data[i, :, :, 0] = inputs
      labels[i, :label_len] = [self.labels[ch] for ch in data[index][1]]
      input_length[i] = data_len
      label_length[i] = label_len
      source_str.append(data[index][1])

    inputs = {'the_input': X_data,
              'the_labels': labels,
              'input_length': input_length,
              'label_length': label_length,
              'source_str': source_str}
    outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
    #print('input_length', input_length, 'label_length', label_length)
    return (inputs, outputs)

  def next_train(self):
    while 1:
      old = self.trainset_index
      ret = self.get_batch(self.minibatch_size, 'train')
      if old > self.trainset_index: # one epoch
        random.shuffle(self.trainset)
      yield ret

  def next_val(self):
    while 1:
      old = self.valset_index
      ret = self.get_batch(self.minibatch_size, 'val')
      if old > self.valset_index: # one epoch
        random.shuffle(self.valset)
      yield ret

  def on_train_begin(self, logs={}):
    pass

  def on_epoch_begin(self, epoch, logs={}):
    pass


# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
  y_pred, labels, input_length, label_length = args
  # the 2 is critical here since the first couple outputs of the RNN
  # tend to be garbage:
  y_pred = y_pred[:, :, :]
  return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch, input_length, ordered_chars):
  out = test_func([word_batch, 0])[0] if test_func else word_batch
  ret = []
  for j in range(out.shape[0]):
    out_best = list(np.argmax(out[j, :input_length[j]], 1))
    #print('out_best', out_best)
    out_best = [k for k, g in itertools.groupby(out_best)]
    # 85 is CTC blank char
    outstr = ''
    for c in out_best:
      if c < output_size-1:
        outstr += ordered_chars[c]
    ret.append(outstr)
  return ret


class VizCallback(keras.callbacks.Callback):

  def __init__(self, test_func, text_img_gen, ordered_chars, output_dir, num_display_words=6):
    self.ordered_chars = ordered_chars
    self.test_func = test_func
    self.output_dir = output_dir
    self.text_img_gen = text_img_gen
    self.num_display_words = num_display_words
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

  def show_edit_distance(self, num):
    num_left = num
    mean_norm_ed = 0.0
    mean_ed = 0.0
    while num_left > 0:
      word_batch = next(self.text_img_gen)[0]
      num_proc = min(word_batch['the_input'].shape[0], num_left)
      decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc], word_batch['input_length'][0:num_proc], self.ordered_chars)
      for j in range(0, num_proc):
        edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
        mean_ed += float(edit_dist)
        mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
      num_left -= num_proc
    mean_norm_ed = mean_norm_ed / num
    mean_ed = mean_ed / num
    print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
          % (num, mean_ed, mean_norm_ed))

  def on_epoch_end(self, epoch, logs={}):
    optimizer = self.model.optimizer
    lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
    print('\nLR: {:.6f}\n'.format(lr))

    self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % epoch))
    self.show_edit_distance(256)
    word_batch = next(self.text_img_gen)[0]
    res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words], word_batch['input_length'][0:self.num_display_words], self.ordered_chars)

    for i in range(self.num_display_words):
      print('Truth = \'%s\' Decoded = \'%s\'' % (word_batch['source_str'][i], res[i]))

# Network parameters
conv_num_filters = [16, 32, 64, 64, 128]
filter_size = 3
pool_sizes = [2, 2, 2]
pool_size = 2*2*2

time_dense_size = 256
rnn_size = 512

def create_model(image_width, image_height, dropout_rest = 0, dropout_gru = 0):
  if K.image_dim_ordering() == 'th':
    input_shape = (1, image_height, image_width)
  else:
    input_shape = (image_height, image_width, 1)

  act = 'relu'
  border_mode = 'same'
  input_data = Input(name='the_input', shape=input_shape, dtype='float32')
  cnn0      = Convolution2D( conv_num_filters[0], 3, 3, border_mode=border_mode, activation='relu', init='he_normal', name='cnn0')(input_data)
  pool0     = MaxPooling2D(pool_size=(2, 2), name='pool0')(cnn0)
  cnn1      = Convolution2D(conv_num_filters[1], 3, 3, border_mode=border_mode, activation='relu', init='he_normal', name='cnn1')(pool0)
  pool1     = MaxPooling2D(pool_size=(2, 2), name='pool1')(cnn1)
  cnn2      = Convolution2D(conv_num_filters[2], 3, 3, border_mode=border_mode, activation='relu', init='he_normal', name='cnn2')(pool1)
  BN0       = BatchNormalization(mode=0, axis=1, name='BN0')(cnn2)
  cnn3      = Convolution2D(conv_num_filters[3], 3, 3, border_mode=border_mode, activation='relu', init='he_normal', name='cnn3')(BN0)
  pool2     = MaxPooling2D(pool_size=(2, 2), name='pool2')(cnn3)
  cnn4      = Convolution2D(conv_num_filters[4], 3, 3, border_mode=border_mode, activation='relu', init='he_normal', name='cnn4')(pool2)
  BN1       = BatchNormalization(mode=0, axis=1, name='BN1')(cnn4)
  inner = BN1

  # inner = Convolution2D(conv_num_filters[0], filter_size, filter_size, border_mode='same',
  #                       activation=act, name='conv1')(input_data)
  # inner = MaxPooling2D(pool_size=(pool_size[0], pool_size[0]), name='max1')(inner)
  # inner = BatchNormalization()(inner)
  # inner = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
  #                       activation=act, name='conv2')(inner)
  # inner = MaxPooling2D(pool_size=(pool_size_2, pool_size_2), name='max2')(inner)
  # inner = BatchNormalization()(inner)
  inner = Permute(dims=(2, 1, 3), name='permute')(inner)

  conv_to_rnn_dims = (image_width / (pool_size), (image_height / (pool_size)) * conv_num_filters[-1])
  inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

  # cuts down input size going into RNN:
  inner = TimeDistributed(Dense(time_dense_size, activation=act, init='he_normal', name='dense1'))(inner)

  inner = Dropout(dropout_rest)(inner)

  # Two layers of bidirecitonal GRUs
  # GRU seems to work as well, if not better than LSTM:
  gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1', dropout_W = dropout_gru, dropout_U = dropout_gru)(inner)
  gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b', dropout_W = dropout_gru, dropout_U = dropout_gru)(inner)
  gru1_merged = merge([gru_1, gru_1b], mode='concat')
  gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2', dropout_W = dropout_gru, dropout_U = dropout_gru)(gru1_merged)
  gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b', dropout_W = dropout_gru, dropout_U = dropout_gru)(gru1_merged)

  # transforms RNN output to character activations:
  inner = TimeDistributed(Dense(output_size, init='he_normal', name='dense2'))(merge([gru_2, gru_2b], mode='concat'))
  inner = Dropout(dropout_rest)(inner)
  y_pred = Activation('softmax', name='softmax')(inner)
  return Model(input=[input_data], output=y_pred)

if __name__ == '__main__':

  ap = argparse.ArgumentParser()
  ap.add_argument("-e", "--epoch", type = int, default = 200,
    help="num of epoch")
  ap.add_argument("-i", "--input", type = str, default = '',
    help="path to the input model")
  ap.add_argument("-o", "--output", required=True, type = str,
    help="output dir")
  ap.add_argument("-l", "--learning", type = float, default = 0.03,
    help="learning rate")
  ap.add_argument("-d", "--decay", type = float, default = 3e-7,
    help="learning rate decay")
  ap.add_argument("--dropout", type = float, default = 0.25,
    help="dropout 1")
  ap.add_argument("--dropout_gru", type = float, default = 0.1,
    help="dropout 2")
  ap.add_argument("--label_file", type = str, default = 'label_file.pkl',
                  help="path to labels")
  ap.add_argument("--train_file", type = str, default = 'train_file.pkl',
                  help="path to trainingset")
  ap.add_argument("--val_file", type = str, default = 'val_file.pkl',
                  help="path to trainingset")
  ap.add_argument("--image_width", type = int, default = 200,
                  help="image width")
  ap.add_argument("--add_noise", type = str, default = 'False',
                  help="whether to add noise to training images")
  ap.add_argument("--min_length", type = int, default = 0,
                  help="minimum length of text to train/val one")
  ap.add_argument("--max_length", type = int, default = 1000,
                  help="maximum length of text to train/val one")


  args = vars(ap.parse_args())

  # Input Parameters
  nb_epoch = args['epoch']
  minibatch_size = 64

  iam = TextImageGenerator(pool_size, minibatch_size, args['image_width'], args['add_noise'] in ['True', 'true'])
  iam.read_iam(args['train_file'], args['val_file'], args['label_file'], args['min_length'], args['max_length'])

  base_model = create_model(args['image_width'], image_height, args['dropout'], args['dropout_gru'])
  if args['input']:
    base_model.load_weights(args['input'])

  input_data = base_model.input
  y_pred = base_model.output

  labels = Input(name='the_labels', shape=[iam.max_line_length], dtype='float32')
  input_length = Input(name='input_length', shape=[1], dtype='int64')
  label_length = Input(name='label_length', shape=[1], dtype='int64')
  # Keras doesn't currently support loss funcs with extra parameters
  # so CTC loss is implemented in a lambda layer
  loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")([y_pred, labels, input_length, label_length])

  lr = args['learning']
  decay = args['decay']
  # clipnorm seems to speeds up convergence
  clipnorm = 10
  sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True, clipnorm=clipnorm)
  #sgd = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

  model = Model(input=[input_data, labels, input_length, label_length], output=[loss_out])
  model.summary()

  # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
  model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

  # captures output of softmax so we can decode the output during visualization
  test_func = K.function([input_data, K.learning_phase()], [y_pred])

  viz_cb = VizCallback(test_func, iam.next_val(), ordered_chars = iam.get_ordered_chars(), output_dir = args['output'])

  model.fit_generator(generator=iam.next_train(), samples_per_epoch=iam.get_num_trainset(),
                      nb_epoch=nb_epoch, validation_data=iam.next_val(), nb_val_samples=iam.get_num_valset(),
                      callbacks=[viz_cb, iam])
