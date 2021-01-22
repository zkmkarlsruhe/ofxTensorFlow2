""" 
Feed-forward network to generate the stylized result.
"""
import tensorflow as tf

class instance_norm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-3):
        super(instance_norm, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.beta = tf.Variable(tf.zeros([input_shape[3]]))
        self.gamma = tf.Variable(tf.ones([input_shape[3]]))

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        x = tf.divide(tf.subtract(inputs, mean), tf.sqrt(tf.add(var, self.epsilon)))
        
        return self.gamma * x + self.beta

class conv_2d(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(conv_2d, self).__init__()
        pad = kernel // 2
        self.paddings = tf.constant([[0, 0], [pad, pad],[pad, pad], [0, 0]])
        self.conv2d = tf.keras.layers.Conv2D(filters, kernel, stride, use_bias=False, padding='valid')
        self.instance_norm = instance_norm()

    def call(self, inputs, relu=True):
        x = tf.pad(inputs, self.paddings, mode='REFLECT')
        x = self.conv2d(x)
        x = self.instance_norm(x)

        if relu:
            x = tf.nn.relu(x)
        return x

class resize_conv_2d(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(resize_conv_2d, self).__init__()
        self.conv = conv_2d(filters, kernel, stride)
        self.instance_norm = instance_norm()
        self.stride = stride

    def call(self, inputs):
        new_h = inputs.shape[1] * self.stride * 2
        new_w = inputs.shape[2] * self.stride * 2
        x = tf.image.resize(inputs, [new_h, new_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = self.conv(x)
        # return x

        """ Redundant """
        x = self.instance_norm(x)

        return tf.nn.relu(x)

class tran_conv_2d(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(tran_conv_2d, self).__init__()
        self.tran_conv = tf.keras.layers.Conv2DTranspose(filters, kernel, stride, padding='same')
        self.instance_norm = instance_norm()

    def call(self, inputs):
        x = self.tran_conv(inputs)
        x = self.instance_norm(x)

        return tf.nn.relu(x)

class residual(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(residual, self).__init__()
        self.conv1 = conv_2d(filters, kernel, stride)
        self.conv2 = conv_2d(filters, kernel, stride)

    def call(self, inputs):
        x = self.conv1(inputs)
        return inputs + self.conv2(x, relu=False)
        

class feed_forward(tf.keras.models.Model):
    def __init__(self):
        super(feed_forward, self).__init__()
        # [filters, kernel, stride]
        self.conv1 = conv_2d(32, 9, 1)     
        self.conv2 = conv_2d(64, 3, 2)           
        self.conv3 = conv_2d(128, 3, 2)     
        self.resid1 = residual(128, 3, 1)         
        self.resid2 = residual(128, 3, 1)          
        self.resid3 = residual(128, 3, 1)     
        self.resid4 = residual(128, 3, 1)     
        self.resid5 = residual(128, 3, 1)    
        #self.tran_conv1 = tran_conv_2d(64, 3, 2)  
        #self.tran_conv2 = tran_conv_2d(32, 3, 2)    
        self.resize_conv1 = resize_conv_2d(64, 3, 2)
        self.resize_conv2 = resize_conv_2d(32, 3, 2)
        self.conv4 = conv_2d(3, 9, 1)              

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resid1(x)
        x = self.resid2(x)
        x = self.resid3(x)
        x = self.resid4(x)
        x = self.resid5(x)
        x = self.resize_conv1(x)
        x = self.resize_conv2(x)
        x = self.conv4(x, relu=False)
        return (tf.nn.tanh(x) * 150 + 255. / 2)     # for better convergence