# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as tk

# Activations

def relu(x):
	return tf.nn.relu(x)

def Relu():
	return tk.layers.ReLU()

def lrelu(x, alpha=0.2):
	return tf.nn.leaky_relu(x, alpha)

def Lrelu(alpha=0.2):
	return tk.layers.LeakyReLU(alpha)

def sigmoid(x):
	return tf.nn.sigmoid(x)

def tanh(x):
	return tf.nn.tanh(x)

# Layers

def Input(input_shape):
	return tk.layers.Input(input_shape)

def Reshape(target_shape):
	return tk.layers.Reshape(target_shape)

def flatten(x):
	return tf.reshape(x, [-1, tf.math.reduce_prod(x.shape[1:])])

def Flatten():
	return tk.layers.Flatten()

def Add():
	return tk.layers.Add()

def Dense(units, activation=None, use_bias=True, sn=False, 
		kernel_initializer='glorot_uniform', bias_initializer='zeros'):
	dense = tk.layers.Dense(units=units, activation=activation, use_bias=use_bias, 
		kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
	return SN(dense) if sn else dense

class Dense_with_w(tk.layers.Layer):
	def __init__(self, activation=None, use_bias=True, sn=False, 
			kernel_initializer='glorot_uniform', bias_initializer='zeros'):
		super(Dense_with_w, self).__init__()
		self.dense = Dense(1, activation=activation, use_bias=use_bias, 
			kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
		self.wrapper = SN(self.dense) if sn else self.dense
		self.use_bias = use_bias

	def call(self, x):
		h = self.wrapper(flatten(x))

		if self.use_bias:
			w = tf.gather(tf.transpose(tf.nn.bias_add(self.dense.kernel, self.dense.bias)), 0)
		else:
			w = tf.gather(tf.transpose(self.dense.kernel), 0)

		return h, w

def Conv2d(filters, kernel_size, strides, padding='same', sn=False, dilation_rate=1, 
		activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
	conv = tk.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, 
		dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, 
		kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
	return SN(conv) if sn else conv

def Deconv2d(filters, kernel_size, strides=2, padding='same', sn=False, dilation_rate=1, 
		activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
	deconv = tk.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, 
		dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, 
		kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
	return SN(deconv) if sn else deconv

def Dropout(rate):
	return tk.layers.Dropout(rate)

class Resblock(tk.layers.Layer):
	def __init__(self, filters=256, norm='in', dropout=False, dropout_rate=0.5):
		super(Resblock, self).__init__()
		self.conv1 = Conv2d(filters, 3, 1)
		self.conv2 = Conv2d(filters, 3, 1)

		if norm == 'bn':
			self.norm1 = BN()
			self.norm2 = BN()
		elif norm == 'in':
			self.norm1 = IN()
			self.norm2 = IN()

		self.dropout = dropout
		if self.dropout:
			self.drop = Dropout(dropout_rate)

	def call(self, x):
		h = relu(self.norm1(self.conv1(x)))
		
		if self.dropout:
			h = self.drop(h)

		return x + self.norm2(self.conv2(h))

class SpadeResblock(tk.layers.Layer):
	def __init__(self, in_filters, out_filters, use_bias=True, sn=False):
		super(SpadeResblock, self).__init__()
		mid_filters = min(in_filters, out_filters)
		self.spade1 = Spade(in_filters, use_bias=use_bias, sn=False)
		self.conv1 = Conv2d(mid_filters, 3, 1, use_bias=use_bias, sn=sn)
		self.spade2 = Spade(mid_filters, use_bias=use_bias, sn=False)
		self.conv2 = Conv2d(out_filters, 3, 1, use_bias=use_bias, sn=sn)
		self.shortcut = False

		if in_filters != out_filters:
			self.shortcut = True
			self.spade3 = Spade(in_filters, use_bias=use_bias, sn=False)
			self.conv3 = Conv2d(out_filters, 1, 1, use_bias=False, sn=sn)

	def call(self, m, x):
		h = self.conv1(lrelu(self.spade1(m, x)))
		h = self.conv2(lrelu(self.spade2(m, h)))
		
		if self.shortcut:
			x = self.conv3(self.spade3(m, x))

		return x + h

class AdaLINResblock(tk.layers.Layer):
	def __init__(self, filters):
		super(AdaLINResblock, self).__init__()
		self.conv1 = Conv2d(filters, 3, 1)
		self.norm1 = AdaLIN()
		self.conv2 = Conv2d(filters, 3, 1)
		self.norm2 = AdaLIN()

	def call(self, x, scale, offset):
		h = relu(self.norm1(self.conv1(x), scale, offset))
		h = self.norm2(self.conv2(h), scale, offset)
		return x + h

# Normalizations

def BN():
	return tk.layers.BatchNormalization()

class IN(tk.layers.Layer):
	def __init__(self):
		super(IN, self).__init__()
		self.epsilon = 1e-5

	def build(self, input_shape):
		self.scale = self.add_weight(name='scale', shape=input_shape[-1:], 
			initializer=tf.random_normal_initializer(1., 0.02), trainable=True)
		self.offset = self.add_weight(name='offset', shape=input_shape[-1:], 
			initializer='zeros', trainable=True)

	def call(self, x):
		mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
		normalized = (x - mean) * tf.math.rsqrt(variance + self.epsilon)
		return self.scale * normalized + self.offset

class LIN(tk.layers.Layer):
	def __init__(self):
		super(LIN, self).__init__()
		self.epsilon = 1e-5

	def build(self, input_shape):
		self.scale = self.add_weight(name='scale', shape=input_shape[-1:], 
			initializer=tf.random_normal_initializer(1., 0.02), trainable=True)
		self.offset = self.add_weight(name='offset', shape=input_shape[-1:], 
			initializer='zeros', trainable=True)
		self.rho = self.add_weight(name='rho', shape=input_shape[-1:], 
			initializer=tf.constant_initializer(0.0), trainable=True, 
			constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))

	def call(self, x):
		in_mean, in_variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
		in_normalized = (x - in_mean) * tf.math.rsqrt(in_variance + self.epsilon)

		ln_mean, ln_variance = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
		ln_normalized = (x - ln_mean) * tf.math.rsqrt(ln_variance + self.epsilon)

		normalized = self.rho * in_normalized + (1 - self.rho) * ln_normalized
		return self.scale * normalized + self.offset

class SN(tk.layers.Wrapper):
	def __init__(self, layer, iteration=1, **kwargs):
		self.iteration = iteration
		super(SN, self).__init__(layer, **kwargs)

	def build(self, input_shape):
		if not self.layer.built:
			self.layer.build(input_shape)
		
		self.w_shape = self.layer.kernel.shape
		self.u = self.add_weight(shape=(1, self.w_shape[-1]), initializer=tf.random_normal_initializer(),
			trainable=False, name='sn_u', dtype=tf.float32)
		
		super(SN, self).build()

	def call(self, inputs, training=True):
		self.update_weights(training)
		output = self.layer(inputs)
		self.restore_weights()
		return output

	def update_weights(self, training):
		self.w = tf.stop_gradient(self.layer.kernel)
		w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
		u_hat = self.u

		for _ in range(self.iteration):
			v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(w_reshaped)))
			u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, w_reshaped))

		if training:
			self.u.assign(u_hat)

		sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
		self.layer.kernel.assign(self.w / sigma)

	def restore_weights(self):
		self.layer.kernel.assign(self.w)

class AdaIN(tk.layers.Layer):
	def __init__(self):
		super(AdaIN, self).__init__()
		self.epsilon = 1e-5

	def call(self, x, scale, offset):
		mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
		normalized = (x - mean) * tf.math.rsqrt(variance + self.epsilon)
		return scale * normalized + offset

class AdaLIN(tk.layers.Layer):
	def __init__(self):
		super(AdaLIN, self).__init__()
		self.epsilon = 1e-5

	def build(self, input_shape):
		self.rho = self.add_weight(name='rho', shape=input_shape[-1:], 
			initializer=tf.constant_initializer(0.9), trainable=True, 
			constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))

	def call(self, x, scale, offset):
		in_mean, in_variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
		in_normalized = (x - in_mean) * tf.math.rsqrt(in_variance + self.epsilon)

		ln_mean, ln_variance = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
		ln_normalized = (x - ln_mean) * tf.math.rsqrt(ln_variance + self.epsilon)

		normalized = self.rho * in_normalized + (1 - self.rho) * ln_normalized
		return scale * normalized + offset

def param_free_norm(x, epsilon=1e-5):
	mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
	std = tf.sqrt(var + epsilon)
	return (x - mean) / std

class Spade(tk.layers.Layer):
	def __init__(self, filters, use_bias=True, sn=False):
		super(Spade, self).__init__()
		self.conv = Conv2d(128, 5, 1, use_bias=use_bias, sn=sn)
		self.gamma = Conv2d(filters, 5, 1, use_bias=use_bias, sn=sn)
		self.beta = Conv2d(filters, 5, 1, use_bias=use_bias, sn=sn)

	def call(self, m, x):
		_, x_h, x_w, _ = x.shape
		_, m_h, m_w, _ = m.shape
		m_down = relu(self.conv(down_sample(m, m_h // x_h, m_w // x_w)))
		return param_free_norm(x) * (1 + self.gamma(m_down)) + self.beta(m_down)

# Losses

def l1_loss(x, y):
	return tf.reduce_mean(tf.abs(x - y))

def l2_loss(x, y):
	return tf.reduce_mean(tf.square(x - y))

def c_loss(labels, logits, binary=True):
	ce = tk.losses.BinaryCrossentropy(from_logits=True) if binary else tk.losses.CategoricalCrossentropy(from_logits=True)
	return ce(labels, logits)

def discriminator_loss(real, fake, gan_type, multi_scale=False):
	if not multi_scale:
		real = [real]
		fake = [fake]

	bce = tk.losses.BinaryCrossentropy(from_logits=True)
	loss = []
	for i in range(len(fake)):
		if gan_type == 'vanilla':
			loss_real = bce(tf.ones_like(real[i]), real[i])
			loss_fake = bce(tf.zeros_like(fake[i]), fake[i])
		elif gan_type == 'lsgan':
			loss_real = tf.reduce_mean(tf.square(real[i] - 1.0))
			loss_fake = tf.reduce_mean(tf.square(fake[i]))
		elif gan_type == 'hinge':
			loss_real = tf.reduce_mean(relu(1.0 - real[i]))
			loss_fake = tf.reduce_mean(relu(1.0 + fake[i]))
		elif gan_type == 'wgan':
			loss_real = -tf.reduce_mean(real[i])
			loss_fake = tf.reduce_mean(fake[i])
		loss.append(loss_real + loss_fake)

	return tf.reduce_mean(loss)

def generator_loss(fake, gan_type, multi_scale=False):
	if not multi_scale:
		fake = [fake]

	bce = tk.losses.BinaryCrossentropy(from_logits=True)
	loss = []
	for i in range(len(fake)):
		if gan_type == 'vanilla':
			loss_fake = bce(tf.ones_like(fake[i]), fake[i])
		elif gan_type == 'lsgan':
			loss_fake = tf.reduce_mean(tf.square(fake[i] - 1.0))
		elif gan_type == 'hinge':
			loss_fake = -tf.reduce_mean(fake[i])
		elif gan_type == 'wgan':
			loss_fake = -tf.reduce_mean(fake[i])
		loss.append(loss_fake)

	return tf.reduce_mean(loss)

def gradient_penalty(D, inter_sample, w_gp=10):
	with tf.GradientTape() as tape:
		tape.watch(inter_sample)
		inter_logit = D(inter_sample, training=True)
	
	grad = tape.gradient(inter_logit, inter_sample)[0]
	norm = tf.norm(flatten(grad), axis=1)
	
	return w_gp * tf.reduce_mean(tf.square(norm - 1.))

def kl_loss(mean, logvar):
	return 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - 1 - logvar)

def feature_loss(real, fake):
	loss = []
	for i in range(len(real)):
		sub_loss = 0
		for j in range(len(real[i]) - 1):
			sub_loss += l1_loss(real[i][j], fake[i][j])
		loss.append(sub_loss)
	return tf.reduce_mean(loss)

def tv_loss(x):
    return l1_loss(x[:, :-1, :, :], x[:, 1:, :, :]) + l1_loss(x[:, :, :-1, :], x[:, :, 1:, :])

# Others

def z_sample(mean, logvar):
	return mean + tf.exp(logvar * 0.5) * tf.random.normal(mean.shape, 0.0, 1.0)

def down_sample(x, scale_factor_h, scale_factor_w) :
	_, h, w, _ = x.shape
	return tf.image.resize(x, [h // scale_factor_h, w // scale_factor_w], method='nearest')

def AvgPool2d(strides=2):
	return tk.layers.AveragePooling2D(pool_size=3, strides=strides, padding='same')

def up_sample(x, scale_factor=2):
	_, h, w, _ = x.shape
	return tf.image.resize(x, [h * scale_factor, w * scale_factor])

def UpSample(size=2):
	return tk.layers.UpSampling2D(size=size)

def global_avg_pooling(x):
    return tf.reduce_mean(x, axis=[1, 2])

def global_max_pooling(x):
    return tf.reduce_max(x, axis=[1, 2])

def lerp_tf(start, end, ratio):
	return start + (end - start) * tf.clip_by_value(ratio, 0.0, 1.0)

def get_pixel_value(img, x, y): # img: N, H, W, C; x: N, H, W; y: N, H, W
	shape = tf.shape(img)
	N, H, W = shape[0], shape[1], shape[2]
	return tf.gather_nd(img, tf.stack([tf.tile(tf.reshape(tf.range(0, N), (N, 1, 1)), (1, H, W)), y, x], 3))

def grid_sample(img, x, y):
	H, W = tf.shape(img)[1], tf.shape(img)[2]
	max_y = tf.cast(H - 1, 'int32')
	max_x = tf.cast(W - 1, 'int32')

	x = 0.5 * ((tf.cast(x, 'float32') + 1.0) * tf.cast(max_x - 1, 'float32'))
	y = 0.5 * ((tf.cast(y, 'float32') + 1.0) * tf.cast(max_y - 1, 'float32'))

	x0 = tf.clip_by_value(tf.cast(tf.floor(x), 'int32'), 0, max_x)
	x1 = tf.clip_by_value(x0 + 1, 0, max_x)
	y0 = tf.clip_by_value(tf.cast(tf.floor(y), 'int32'), 0, max_y)
	y1 = tf.clip_by_value(y0 + 1, 0, max_y)

	Ia = get_pixel_value(img, x0, y0)
	Ib = get_pixel_value(img, x0, y1)
	Ic = get_pixel_value(img, x1, y0)
	Id = get_pixel_value(img, x1, y1)

	x0 = tf.cast(x0, 'float32')
	x1 = tf.cast(x1, 'float32')
	y0 = tf.cast(y0, 'float32')
	y1 = tf.cast(y1, 'float32')

	wa = tf.expand_dims((x1 - x) * (y1 - y), -1)
	wb = tf.expand_dims((x1 - x) * (y - y0), -1)
	wc = tf.expand_dims((x - x0) * (y1 - y), -1)
	wd = tf.expand_dims((x - x0) * (y - y0), -1)

	return tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

def gram_matrix(x):
	N, H, W, C = x.shape
	x = tf.reshape(x, (N, H * W, C))
	return tf.matmul(tf.transpose(x, (0, 2, 1)), x) / (H * W * C)

def gram_mse(x, y):
	return l2_loss(gram_matrix(x), gram_matrix(y))

def roi_align(feature, boxes, indices, img_size, output_size):
	output_size[0] = output_size[0] * 2
	output_size[1] = output_size[1] * 2

	x0, y0, x1, y1 = tf.split(boxes, 4, axis=-1)
	binW = (x1 - x0) / output_size[1]
	binH = (y1 - y0) / output_size[0]

	nx0 = (x0 + binW / 2 - 0.5) / (img_size[1] - 1)
	ny0 = (y0 + binH / 2 - 0.5) / (img_size[0] - 1)
	nW = binW * (output_size[1] - 1) / (img_size[1] - 1)
	nH = binH * (output_size[0] - 1) / (img_size[0] - 1)

	new_boxes = tf.concat([ny0, nx0, ny0 + nH, nx0 + nW], axis=-1)
	sampled = tf.image.crop_and_resize(feature, new_boxes, indices, output_size)

	return tf.nn.avg_pool2d(sampled, 2, 2, padding='VALID')