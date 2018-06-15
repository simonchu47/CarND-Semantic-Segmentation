import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import tensorflow as tf
import scipy.misc

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

model_save = "saved_model_20"
vgg_input_tensor_name = 'image_input:0'
vgg_seg_output_tensor_name = 'seg_final_output:0'
vgg_keep_prob_tensor_name = 'keep_prob:0'
num_classes = 3
image_shape = (608, 800)
#original_image_shape = (600, 800)

if tf.saved_model.loader.maybe_saved_model_directory(model_save):
    print("Check saved model OK")
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ["vgg16_semantic"], model_save)
        graph = tf.get_default_graph()
        image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
        keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        nn_last_layer = graph.get_tensor_by_name(vgg_seg_output_tensor_name)
        logits = tf.reshape(nn_last_layer, [-1, num_classes])
        
        #sess.run(tf.global_variables_initializer())
        
        
        print("Load saved model OK")

        
        for rgb_frame in video:
            original_image_shape = (rgb_frame.shape[0], rgb_frame.shape[1])
            rgb_frame_scaled = scipy.misc.imresize(rgb_frame, image_shape)
            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_input: [rgb_frame_scaled]})

            #print("im_softmax[0] shape is {}".format(im_softmax[0].shape))
            road_result = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            car_result = im_softmax[0][:, 2].reshape(image_shape[0], image_shape[1])

            road_seg = (road_result > 0.5).reshape(image_shape[0], image_shape[1], 1)
            car_seg = (car_result > 0.5).reshape(image_shape[0], image_shape[1], 1)
            
            road_mask = np.dot(road_seg, np.array([[0, 255, 0, 127]]))
            car_mask = np.dot(car_seg, np.array([[0, 255, 0, 127]]))
            
            road_mask = scipy.misc.toimage(road_mask, mode="RGBA")
            car_mask = scipy.misc.toimage(car_mask, mode="RGBA")
            
            road_im = scipy.misc.toimage(rgb_frame_scaled)
            road_im.paste(road_mask, box=None, mask=road_mask)
            road_im = scipy.misc.imresize(road_im, original_image_shape)
            
            car_im = scipy.misc.toimage(rgb_frame_scaled)
            car_im.paste(car_mask, box=None, mask=car_mask)
            car_im = scipy.misc.imresize(car_im, original_image_shape)
            
            road_im_path = "road_" + str(frame) + ".png"
            car_im_path = "car_" + str(frame) + ".png"
            
            scipy.misc.imsave(road_im_path, np.array(road_im))
            scipy.misc.imsave(car_im_path, np.array(car_im))
            
            #print("road_result size is {}".format(road_result.shape))
            #print("car_result size is {}".format(car_result.shape))

            #binary_road_result = np.where(road_result>0.5,1,0).astype('uint8')
            #binary_car_result = np.where(car_result>0.5,1,0).astype('uint8')

            #print("binary_road_result size is {}".format(binary_road_result.shape))
            #print("binary_car_result size is {}".format(binary_car_result.shape))
            #binary_road_result = scipy.misc.imresize(binary_road_result, original_image_shape)
            #binary_car_result = scipy.misc.imresize(binary_car_result, original_image_shape)
            #print("binary_road_result size is {}".format(binary_road_result.shape))
            #print("binary_car_result size is {}".format(binary_car_result.shape))
            #answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
            
            # Increment frame
            frame+=1

# Print output in proper json format
#print (json.dumps(answer_key))
