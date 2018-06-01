import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import tensorflow as tf

file = sys.argv[-1]

if file == 'demo.py':
  print ("Error loading video")
  quit

# Define encoder function
def encode(array):
        pil_img = Image.fromarray(array)
        buff = BytesIO()
        pil_img.save(buff, format="PNG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")

video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

model_save = "saved_model_200"
vgg_input_tensor_name = 'image_input:0'
vgg_seg_output_tensor_name = 'seg_final_output:0'
vgg_keep_prob_tensor_name = 'keep_prob:0'
num_classes = 3

if tf.saved_model.loader.maybe_saved_model_directory(model_save):
    print("Check saved model OK")
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ["vgg16_semantic"], model_save)
        graph = tf.get_default_graph()
        image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
        nn_last_layer = graph.get_tensor_by_name(vgg_seg_output_tensor_name)
        keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        logits = tf.reshape(nn_last_layer, [-1, num_classes])
        
        sess.run(tf.global_variables_initializer())
        
        
        print("Load saved model OK")

        
        for rgb_frame in video:
            
            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_input: [rgb_frame]})
            
            """
            # Grab red channel
            red = rgb_frame[:,:,0]
            # Look for red cars :)
            binary_car_result = np.where(red>250,1,0).astype('uint8')

            # Look for road :)
            binary_road_result = binary_car_result = np.where(red<20,1,0).astype('uint8')

            answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
            """
            
            # Increment frame
            frame+=1

# Print output in proper json format
print (json.dumps(answer_key))
