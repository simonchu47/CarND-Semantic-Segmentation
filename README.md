# Semantic Segmentation
### Introduction
This project is the elective project of Udacity Self-Driving Car Nanodegree Term3. In this project, each pixel in images will be labeled as of a road or not using a Fully Convolutional Network (FCN).

### Setup
Here is a [guide](https://github.com/simonchu47/CarND-Semantic-Segmentation/blob/master/HOWTO.md) showing how to prepare the environment.

### Start
##### Implement
###### 1. Loading the pretrained VGG-16 model into TensorFlow

```python
def load_vgg(sess, vgg_path):    vgg_tag = 'vgg16'
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
```
##### 2. Creating the layers for a fully convolutional network, including the skip-layers

```python
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, strides=(2, 2), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='vgg_layer4_out_scaled')
    pool4 = tf.layers.conv2d(vgg_layer4_out_scaled, num_classes, 1, strides=(1,1), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output_1 = tf.add(output, pool4)
    output_2 = tf.layers.conv2d_transpose(output_1, num_classes, 4, strides=(2, 2), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='vgg_layer3_out_scaled')
    pool_3 = tf.layers.conv2d(vgg_layer3_out_scaled, num_classes, 1, strides=(1,1), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output_3 = tf.add(output_2, pool_3)
    output_4 = tf.layers.conv2d_transpose(output_3, num_classes, 16, strides=(8, 8), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        
    return output_4
```    
##### 3. Building the TensorFLow loss and optimizer operations

```python
def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    logits = tf.reshape(nn_last_layer, [-1, num_classes])
    logits_labels = tf.reshape(correct_label, [-1, num_classes])
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=logits_labels, logits=logits)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.01
    cross_entropy_loss = cross_entropy_loss + reg_constant*sum(reg_losses)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0.9, beta2 = 0.999)
    train_op = optimizer.minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss    
```
##### 4. Defining the training steps

```python
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    for epoch in range(epochs):
        print("Epoch {}...".format(epoch))
        for image, label in get_batches_fn(batch_size): 
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0001})
    
```
##### 5. Defining the pipe line

```python
def run():
    with tf.Session() as sess:
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)
```

##### Run
Run the following command to run the project:
```
python main.py
```

### Notice

1. Branch ``lyft_challenge`` is created for Lyft Challenge.
- [Here](https://github.com/simonchu47/CarND-Semantic-Segmentation-1) is the forked repository from AsadZia, who got the No.1 rank of Lyft Challenge.
- This project only focus on FCN based on VGG16, but there is still other state of the art developed for semantic segmentation. [Here](https://github.com/simonchu47/TFSegmentation) is the forked repositoy from Msiam/TFSegmentation. I've modified the repository for training on the data set of Lyft Challenge. Therefore we can compare the performance between the models which come from the combinations of different encoders and decoders.
 
