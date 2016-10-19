
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

from handwriting_ocr import *

np.random.seed(55)

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

K.set_learning_phase(0)

weights_file = sys.argv[1]
image_file = sys.argv[2]
truth = sys.argv[3]

ordered_chars = get_ordered_chars()
ordered_chars[-1] = ''
inputs = read_image(image_file, False, True).astype(np.float32)
image_height, image_width = inputs.shape[:2]
inputs = np.expand_dims(inputs, 0)
inputs = np.expand_dims(inputs, 3)

with K.tf.Session() as sess:
  model = create_model(image_width, image_height)
  model.load_weights(weights_file)
  print model.output
  print model.input

  probability = sess.run(model.output, {model.input: inputs})
  #print probability
  sentence = decode_batch(None, probability, ordered_chars)[0]
  edit_dist = editdistance.eval(sentence, truth)
  print truth
  print sentence
  print edit_dist

#K.tf.ConfigProto
