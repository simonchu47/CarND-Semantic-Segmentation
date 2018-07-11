#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 23:13:20 2018

@author: simon
"""
import tensorflow as tf
import sys, json, base64
import skvideo.io
import scipy.misc
import timeit

def main():
    
    nn_shape = (192, 256)
    
    file = sys.argv[-1]
    myname = __file__
    if file == myname:
        print ("Error loading video")
        exit()
    
    video = skvideo.io.vread(file)
    
    with tf.gfile.GFile('./frozen_graph.pb', 'rb') as f:
        graph_def_optimized = tf.GraphDef()
        graph_def_optimized.ParseFromString(f.read())

    G = tf.Graph()
    
    with tf.Session(graph=G) as sess:
        logits, = tf.import_graph_def(graph_def_optimized, return_elements=['seg_final_output:0'])
        image_input = G.get_tensor_by_name('import/image_input:0')
        keep_prob = G.get_tensor_by_name('import/keep_prob:0')

        sess.run(tf.global_variables_initializer())

        start_time = timeit.default_timer()
        
        for rgb_frame in video:
            original_image_shape = (rgb_frame.shape[0], rgb_frame.shape[1])
            rgb_frame_scaled = scipy.misc.imresize(rgb_frame, nn_shape)
            seg_map = sess.run(
                [logits],
                {keep_prob: 1.0, image_input: [rgb_frame_scaled]})
        
        elapsed = timeit.default_timer() - start_time
        print("inferencing time is {}".format(elapsed))

if __name__ == '__main__':
    main()
    
                                      
                                      