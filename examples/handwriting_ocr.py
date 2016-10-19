'''This example uses a convolutional stack followed by a recurrent stack
and a CTC logloss function to perform optical character recognition
of generated text images. I have no evidence of whether it actually
learns general shapes of text, or just is able to recognize all
the different fonts thrown at it...the purpose is more to demonstrate CTC
inside of Keras.  Note that the font list may need to be updated
for the particular OS in use.

This starts off with 4 letter words. After 10 or so epochs, CTC
learns translational invariance, so longer words and groups of words
with spaces are gradually fed in.  This gradual increase in difficulty
is handled using the TextImageGenerator class which is both a generator
class for test/train data and a Keras callback class. Every 10 epochs
the wordlist that the generator draws from increases in difficulty.

The table below shows normalized edit distance values. Theano uses
a slightly different CTC implementation, so some Theano-specific
hyperparameter tuning would be needed to get it to match Tensorflow.

            Norm. ED
Epoch |   TF   |   TH
------------------------
    10   0.072    0.272
    20   0.032    0.115
    30   0.024    0.098
    40   0.023    0.108

This requires cairo and editdistance packages:
pip install cairocffi
pip install editdistance

Due to the use of a dummy loss function, Theano requires the following flags:
on_unused_input='ignore'

Created by Mike Henry
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

np.random.seed(55)

image_height = 64 #128
image_width = 1024 # int(sys.argv[1]) # 1024
output_size = 85

trimBg = Image.new('L', (3000,1000), 255)
inputAvg = 255/2.
inputStd = 255/2.


def speckle(img):
  severity = np.random.uniform(0, 0.6)
  blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
  img_speck = (img + (blur * (img != 1)))
  img_speck[img_speck > 1] = 1
  img_speck[img_speck <= 0] = 0
  return img_speck

def read_image(imagePath, isTrain, keep_aspect = False):
  # Load all stroke points for the current line as inputs
  image = Image.open(imagePath)
  if isTrain:
    image = PIL.ImageOps.invert(image)
    image = image.rotate(np.random.uniform(-5, 5))
    image = PIL.ImageOps.invert(image)
  image = image.crop(ImageChops.difference(image, trimBg).getbbox())
  imageWidth = int(image.size[0] / float(image.size[1]) * image_height)
  #self.max_image_width = max(self.max_image_width, imageWidth)
  width = imageWidth if keep_aspect else image_width
  image = image.resize((width, image_height))
  inputs = (np.array(image) - inputAvg) / inputStd
  # if isTrain:
  #   inputs = speckle(inputs)
  return inputs

# Uses generator functions to supply train/test with
# data. Image renderings are text are created on the fly
# each time with random perturbations

class TextImageGenerator(keras.callbacks.Callback):

  def __init__(self, downsample_width, minibatch_size):

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
    return len(self.labels) + 1

  def get_num_trainset(self):
    return len(self.trainset)

  def get_num_valset(self):
    return len(self.valset)

  def get_ordered_chars(self):
    chars = sorted(self.labels.items(), key = lambda x: x[1])
    return [x[0] for x in chars] + [' ']

  def read_one_dataset(self, dataset):
    print('reading %s' % dataset)
    data = []
    # Process all active samples in the current training set
    for sample in file(dataset + '.txt').readlines():
      basePath = re.sub(r'^(((\w+)-\d+)\w?)$', '\g<3>/\g<2>/\g<1>', sample.strip())

      # Check whether the sample files exist
      asciiPath = 'ascii/%s.txt' % basePath
      if not os.path.exists(asciiPath):
        print 'ERROR: Sample %s does not exist!' % basePath
        continue

      #print "Processing sample", basePath

      # Read the plain text of the sample
      with open(asciiPath, 'r') as asciiFile:
        text = re.sub(r'.*[\r\n]+CSR:\s*[\r\n]+', '', asciiFile.read(), 1, re.DOTALL)
        lines = re.split(r'[\r\n]+', text.strip())

        # Process each line in the text separately
        for i in range(0, len(lines)):
          line = lines[i].strip()
          self.max_line_length = max(self.max_line_length, len(line))
          imagePath = 'lineImages/%s-%02d.tif' % (basePath, i + 1)
          if not os.path.exists(imagePath):
            print 'ERROR: Image %s do not exist!' % imagePath
            continue

          # Process all characters and build the target sequence
          label = [self.labels[ch] for ch in line]

          data.append((imagePath, label, line))
    rounded_len = len(data) / self.minibatch_size * self.minibatch_size
    return data[:rounded_len]

  # num_words can be independent of the epoch size due to the use of generators
  # as max_string_len grows, num_words can grow
  def read_iam(self):

    self.labels = {}
    print "Loading labels"
    for fileName in glob.glob('ascii/*/*/*.txt'):
      asciiFile = open(fileName, 'r')
      text = re.sub(r'.*[\r\n]+CSR:\s*[\r\n]+', '', asciiFile.read(), 1, re.DOTALL)
      for char in text:
        if char == '\n' or char == '\r': continue
        if char not in self.labels:
          self.labels[char] = len(self.labels)
        asciiFile.close()
    assert(len(self.labels)+1 == output_size)

    self.trainset = read_one_dataset('trainset')
    self.valset = self.read_one_dataset('testset_v') + self.read_one_dataset('testset_t')
    self.testset = self.read_one_dataset('testset_f')
    print('iam: max_line_length %s, max_image_width %s' % (self.max_line_length, self.max_image_width))

  # each time an image is requested from train/val/test, a new random
  # painting of the text is performed
  def get_batch(self, size, train):
    if K.image_dim_ordering() == 'th':
      X_data = np.ones([size, 1, image_height, image_width])
    else:
      X_data = np.ones([size, image_height, image_width, 1])

    labels = np.ones([size, self.max_line_length])
    input_length = np.zeros([size, 1])
    label_length = np.zeros([size, 1])
    source_str = []

    for i in range(0, size):
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

      inputs = self.read_image(data[index][0], train == 'train')
      if K.image_dim_ordering() == 'th':
        X_data[i, 0, :, :] = inputs
      else:
        X_data[i, :, :, 0] = inputs
      label_len = len(data[index][2])
      labels[i, :label_len] = data[index][1]
      input_length[i] = self.downsample_width
      label_length[i] = label_len
      source_str.append(data[index][2])

    inputs = {'the_input': X_data,
              'the_labels': labels,
              'input_length': input_length,
              'label_length': label_length,
              'source_str': source_str}
    outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
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

def decode_batch(test_func, word_batch, ordered_chars):
  out = test_func([word_batch, 0])[0] if test_func else word_batch
  ret = []
  for j in range(out.shape[0]):
    out_best = list(np.argmax(out[j, :], 1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    # 26 is space, 27 is CTC blank char
    outstr = ''
    for c in out_best:
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
    os.makedirs(self.output_dir)

  def show_edit_distance(self, num):
    num_left = num
    mean_norm_ed = 0.0
    mean_ed = 0.0
    while num_left > 0:
      word_batch = next(self.text_img_gen)[0]
      num_proc = min(word_batch['the_input'].shape[0], num_left)
      decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc], self.ordered_chars)
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
    self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % epoch))
    self.show_edit_distance(256)
    word_batch = next(self.text_img_gen)[0]
    res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words], self.ordered_chars)

    for i in range(self.num_display_words):
      print('Truth = \'%s\' Decoded = \'%s\'' % (word_batch['source_str'][i], res[i]))

# Network parameters
conv_num_filters = 16
filter_size = 3
pool_size_1 = 4
pool_size_2 = 2
time_dense_size = 32
rnn_size = 512

def create_model(image_width, image_height, dropout1 = 0, dropout2 = 0):
  if K.image_dim_ordering() == 'th':
    input_shape = (1, image_height, image_width)
  else:
    input_shape = (image_height, image_width, 1)

  act = 'relu'
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

  inner = Dropout(dropout1)(inner)

  # Two layers of bidirecitonal GRUs
  # GRU seems to work as well, if not better than LSTM:
  gru_1 = GRU(rnn_size, return_sequences=True, name='gru1', dropout_W = dropout2, dropout_U = dropout2)(inner)
  gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, name='gru1_b', dropout_W = dropout2, dropout_U = dropout2)(inner)
  gru1_merged = merge([gru_1, gru_1b], mode='sum')
  gru_2 = GRU(rnn_size, return_sequences=True, name='gru2', dropout_W = dropout2, dropout_U = dropout2)(gru1_merged)
  gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, dropout_W = dropout2, dropout_U = dropout2)(gru1_merged)

  # transforms RNN output to character activations:
  inner = TimeDistributed(Dense(output_size, name='dense2'))(merge([gru_2, gru_2b], mode='concat'))
  y_pred = Activation('softmax', name='softmax')(inner)
  return Model(input=[input_data], output=y_pred)

if __name__ == '__main__':

  ap = argparse.ArgumentParser()
  ap.add_argument("-e", "--epoch", type = int, default = 200,
    help="num of epoch")
  ap.add_argument("-i", "--input", type = string,
    help="path to the input model")
  ap.add_argument("-o", "--output", required=False, type = string,
    help="output dir")
  ap.add_argument("-l", "--learning", type = float, default = 0.03,
    help="learning rate")
  ap.add_argument("-d", "--decay", type = float, default = 3e-7,
    help="learning rate decay")
  ap.add_argument("--dropout1", type = float, default = 0.25,
    help="dropout 1")
  ap.add_argument("--dropout2", type = float, default = 0.1,
    help="dropout 2")
  args = vars(ap.parse_args())

  # Input Parameters
  nb_epoch = args['epoch']
  minibatch_size = 64

  time_steps = image_width / (pool_size_1 * pool_size_2)
  iam = TextImageGenerator(time_steps, minibatch_size)
  iam.read_iam()

  base_model = create_model(image_width, image_height, args['dropout1'], args['dropout2'])
  if 'input' in args:
    base_model.load_weights(args['input'])

  input_data = base_model.input[0]
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
  clipnorm = 5
  sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True, clipnorm=clipnorm)

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
