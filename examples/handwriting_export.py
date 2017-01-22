
import os
import itertools
import re
import time
import datetime
import random
import editdistance
import numpy as np
import math, glob, sys, os, re
import argparse
import tensorflow as tf
from tensorflow.python.framework import tensor_shape, graph_util
from tensorflow.python.platform import gfile

from handwriting_word_ocr import *

np.random.seed(55)

K.set_learning_phase(0)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, type = str,
  help="path to the input model")
ap.add_argument("-o", "--output", required=True, type = str,
  help="output dir")
ap.add_argument("--width", type = int, default = 456,
  help="width")
args = vars(ap.parse_args())

with K.tf.Session() as sess:
  model = create_model(args['width'], image_height, 0.0, 0.0)
  model.load_weights(args['input'])
  print model.output
  print model.input

  graph = sess.graph
  rc = tf.constant((1, -1, 128), name ='Reshape_2/modified_shape')
  nrc = graph.get_tensor_by_name('Reshape_2/modified_shape:0')
  graph._nodes_by_name['Reshape_2']._update_input(1, nrc)

  # 7296 / 128 = 57 steps
  # 1824 / 57 = 32
  rc = tf.constant((1, -1, 32), name ='Reshape_4/modified_shape')
  nrc = graph.get_tensor_by_name('Reshape_4/modified_shape:0')
  graph._nodes_by_name['Reshape_4']._update_input(1, nrc)

  # 29184/ 57 = 512
  rc = tf.constant((1, -1, 512), name ='Reshape_6/modified_shape')
  nrc = graph.get_tensor_by_name('Reshape_6/modified_shape:0')
  for num in [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]:
    graph._nodes_by_name['Reshape_%s' % num]._update_input(1, nrc)

  # 4845 / 57 = 85
  rc = tf.constant((1, -1, 85), name ='Reshape_30/modified_shape')
  nrc = graph.get_tensor_by_name('Reshape_30/modified_shape:0')
  graph._nodes_by_name['Reshape_30']._update_input(1, nrc)

  output_graph_def = graph_util.convert_variables_to_constants(
    sess, sess.graph.as_graph_def(), ['div'])
  with gfile.FastGFile(os.path.join(args['output'], 'english_output_graph.pb'), 'wb') as f:
    f.write(output_graph_def.SerializeToString())
