import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
        
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), padding='same',
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    output = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, strides=(2, 2), padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='vgg_layer4_out_scaled')

    pool4 = tf.layers.conv2d(vgg_layer4_out_scaled, num_classes, 1, strides=(1,1), padding='same',
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    output_1 = tf.add(output, pool4)

    output_2 = tf.layers.conv2d_transpose(output_1, num_classes, 4, strides=(2, 2), padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='vgg_layer3_out_scaled')

    pool_3 = tf.layers.conv2d(vgg_layer3_out_scaled, num_classes, 1, strides=(1,1), padding='same',
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    output_3 = tf.add(output_2, pool_3)

    output_4 = tf.layers.conv2d_transpose(output_3, num_classes, 16, strides=(8, 8), padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        
    return output_4
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, [-1, num_classes])
    logits_labels = tf.reshape(correct_label, [-1, num_classes])
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=logits_labels, logits=logits)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.01
    cross_entropy_loss = cross_entropy_loss + reg_constant*sum(reg_losses)
    #beta1 = tf.Variable(0.9)
    #beta2 = tf.Variable(0.999)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0.9, beta2 = 0.999)
    train_op = optimizer.minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    # need to initialize local variables for this to run `tf.metrics.mean_iou`
    #sess.run(tf.local_variables_initializer())

    for epoch in range(epochs):
        print("Epoch {}...".format(epoch))
        for image, label in get_batches_fn(batch_size): 
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0001})
            #sess.run(iou_op)
            #print("Mean IoU =", sess.run(iou))
            print("The loss is {}".format(loss))
            
tests.test_train_nn(train_nn)

def mean_iou(ground_truth, prediction, num_classes):
    # TODO: Use `tf.metrics.mean_iou` to compute the mean IoU.
    iou, iou_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes)
    return iou, iou_op

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    learning_rate = 0.001
    epochs = 200
    batch_size = 16

    correct_label = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], 2))
    input_image = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], 3))
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())

        #sess.run(tf.initialize_all_variables()) 

        #sess.run(sess.graph.get_tensor_by_name('beta1_power/Assign:0'))
        #sess.run(sess.graph.get_tensor_by_name('beta2_power/Assign:0'))

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
       
        #iou, iou_op = mean_iou(correct_label, logits, num_classes)  
        
        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        model_save = "saved_model_" + str(epochs)
        builder = tf.saved_model.builder.SavedModelBuilder(model_save)
        builder.add_meta_graph_and_variables(sess, ["vgg16_semantic"])
        builder.save()
        
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
