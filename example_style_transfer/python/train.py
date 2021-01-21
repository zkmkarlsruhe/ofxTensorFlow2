import tensorflow as tf
import numpy as np
import os

from modules.utils import tensor_to_image, load_img, create_folder, clip_0_1
from modules.vgg19 import preprocess_input, VGG19
from modules.forward import feed_forward


def vgg_layers(layer_names):
    vgg = VGG19(include_top = False, weights = 'imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(features, normalize = True):
    batch_size , height, width, filters = features.shape
    features = tf.reshape(features, (batch_size, height*width, filters))

    tran_f = tf.transpose(features, perm=[0,2,1])
    gram = tf.matmul(tran_f, features)
    if normalize:
        gram /= tf.cast(height*width, tf.float32)

    return gram

def style_loss(style_outputs, style_target):
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_target[name])**2)
                        for name in style_outputs.keys()])

    return style_loss / len(style_outputs)

def content_loss(content_outputs, content_target):
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_target[name])**2)
                            for name in content_outputs.keys()])

    return content_loss / len(content_outputs)

def total_variation_loss(img):
    x_var = img[:,:,1:,:] - img[:,:,:-1,:]
    y_var = img[:,1:,:,:] - img[:,:-1,:,:]

    return tf.reduce_mean(tf.square(x_var)) + tf.reduce_mean(tf.square(y_var))

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        preprocessed_input = preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        # Compute the gram_matrix
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        # Features that extracted by VGG
        style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}
        content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}

        return {'content':content_dict, 'style':style_dict}


def trainer(style_file, dataset_path, weights_path, content_weight, style_weight, 
            tv_weight, learning_rate, batch_size, epochs, debug):

    # Setup the given layers
    content_layers = ['block4_conv2']

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    
    # Build Feed-forward transformer

    network = feed_forward()

    
    # Build VGG-19 Loss network
    extractor = StyleContentModel(style_layers, content_layers)

    # Load style target image
    style_image = load_img(style_file, resize=False)

    # Initialize content target images
    batch_shape = (batch_size, 256, 256, 3)
    X_batch = np.zeros(batch_shape, dtype=np.float32)

    # Extract style target 
    style_target = extractor(style_image*255.0)['style']

    # Build optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss_metric = tf.keras.metrics.Mean()
    sloss_metric = tf.keras.metrics.Mean()
    closs_metric = tf.keras.metrics.Mean()
    tloss_metric = tf.keras.metrics.Mean()


    @tf.function()
    def train_step(X_batch):
        with tf.GradientTape() as tape:

            content_target = extractor(X_batch*255.0)['content']
            image = network(X_batch)
            outputs = extractor(image)
            
            s_loss = style_weight * style_loss(outputs['style'], style_target)
            c_loss = content_weight * content_loss(outputs['content'], content_target)
            t_loss = tv_weight * total_variation_loss(image)
            loss = s_loss + c_loss + t_loss

        grad = tape.gradient(loss, network.trainable_variables)
        opt.apply_gradients(zip(grad, network.trainable_variables))

        loss_metric(loss)
        sloss_metric(s_loss)
        closs_metric(c_loss)
        tloss_metric(t_loss)


    train_dataset = tf.data.Dataset.list_files(dataset_path + '/*.jpg')
    train_dataset = train_dataset.map(load_img,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(1024)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    import time
    start = time.time()


    @tf.function(input_signature=[tf.TensorSpec([None, 640, 480, 3], dtype=tf.float32)])
    def model_predict(input_1):
        return {'outputs': network(input_1, training=False)}


    for e in range(epochs):
        print('Epoch {}'.format(e))
        iteration = 0

        for img in train_dataset:

            for j, img_p in enumerate(img):
                X_batch[j] = img_p

            iteration += 1
            
            train_step(X_batch)

            if e == 0 and iteration == 0:
                network.summary()

            if iteration % 100 == 0:
                # Save model
                network.save('../bin/data/model'+str(iteration), signatures={'serving_default': model_predict})
                print('=====================================')
                print('            Model saved!            ')
                print('=====================================\n')
                
                # network.save_weights("model_check", save_format='tf')

                print('step %s: loss = %s' % (iteration, loss_metric.result()))
                print('s_loss={}, c_loss={}, t_loss={}'.format(sloss_metric.result(), closs_metric.result(), tloss_metric.result()))

    end = time.time()
    print("Total time: {:.1f}".format(end-start))
    