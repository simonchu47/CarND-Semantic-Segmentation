#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 23:24:46 2018

@author: simon
"""
import sys, skvideo.io, json, base64
#import numpy as np
#from PIL import Image
#from io import BytesIO, StringIO
import tensorflow as tf
#import scipy.misc

file = sys.argv[-1]

def load_graph(graph_file, use_xla=False):
    jit_level = 0
    config = tf.ConfigProto()
    if use_xla:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        ops = sess.graph.get_operations()
        n_ops = len(ops)
        return sess.graph, ops
    
sess, base_ops = load_graph(file)
print(len(base_ops))