
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

from handwriting_word_ocr import read_image, create_model, image_height, pool_size

np.random.seed(55)

#K.tf.ConfigProto

if __name__ == '__main__':

  ap = argparse.ArgumentParser()
  ap.add_argument("--label_file", type = str, default = 'label_file.pkl',
                  help="path to labels")
  ap.add_argument("--model_file", type = str,
                  help="path to trainingset")
  ap.add_argument("--image_file", type = str,
                  help="path to trainingset")
  ap.add_argument("--image_width", type = int, default = 600
                  help="path to trainingset")

  args = vars(ap.parse_args())

  K.set_learning_phase(0)

# assuming image is alread black (0) /white (255)
  bool_arr = np.asarray(Image.open(args['image_file']).convert('L')) < 128
  input_img, width = read_image(bool_arr, False, False, bool_arr, args['image_width'])
  input_img = np.expand_dims(input_img, 0)
  input_img = np.expand_dims(input_img, 3)

  labels = pickle.load(open(args['label_file']))
  ordered_chars = [x[0] for x in sorted(labels.items(), key = lambda x: x[1])]


  with K.tf.Session() as sess:
    model = create_model(args['image_width'], image_height, 0.0, 0.0)
    model.load_weights(args['model_file'])
    print model.output
    print model.input

    probability = sess.run(model.output, {model.input: inputs})

    sentence = decode_batch(None, probability, input_length = np.array([int(math.ceil(width / pool_size))]), ordered_chars)[0]
    print sentence
