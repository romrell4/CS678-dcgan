from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *


class DCGAN(object):
	def __init__(self, sess, config):

		self.training = config.training
		self.crop     = config.crop
		self.c        = config.c
		self.sess     = sess

		self.chk_dir = config.chk_dir
		self.smp_dir = config.smp_dir
		self.dbname  = config.dbname

		self.batch_size = config.batch_size
		self.sample_num = config.batch_size

		self.gfc_dim = self.dfc_dim = 1024
		self.gf_dim  = self.df_dim  = 64
		self.z_dim   = config.test_size

		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')
		self.g_bn0 = batch_norm(name='g_bn0')
		self.g_bn1 = batch_norm(name='g_bn1')
		self.g_bn2 = batch_norm(name='g_bn2')

		if self.dbname == 'mnist':
			self.in_dim, self.out_dim, self.y_dim = 28, 28, 10
			self.data_X, self.data_y = self.load_mnist()
			self.c_dim = self.data_X[0].shape[-1]

		else:
			self.in_dim, self.out_dim, self.y_dim = 108, 64, None
			self.data = glob(os.path.join("./data", self.dbname, "*.jpg"))
			shape = imread(self.data[0]).shape
			self.c_dim = 1 if len(shape) < 3 else shape[-1]
			self.d_bn3 = batch_norm(name='d_bn3')
			self.g_bn3 = batch_norm(name='g_bn3')

		self.grayscale = (self.c_dim == 1)

		self.build_model()

		if self.training:
			self.train(config)
		else:
			if not self.load():
				raise Exception("[!] Train a model first, then run test mode")


	def build_model(self):

		if self.y_dim:
			self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
		else:
			self.y = None

		if self.crop:
			image_dims = [self.out_dim, self.out_dim, self.c_dim]
		else:
			image_dims = [self.in_dim, self.in_dim, self.c_dim]

		self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
		self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
		self.z_sum = tf.summary.histogram("z", self.z)

		self.G                  = self.generator(self.z, self.y)
		self.D, self.D_logits   = self.discriminator(self.inputs, self.y, reuse=False)
		self.sampler            = self.sampler(self.z, self.y)
		self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
		
		self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D), logits=self.D_logits))
		self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_), logits=self.D_logits_))
		self.g_loss      = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_), logits=self.D_logits_))

		self.d_loss = self.d_loss_real + self.d_loss_fake

		t_vars = tf.trainable_variables()

		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		self.saver = tf.train.Saver()


	def train(self, config):

		d_opt = tf.train.AdamOptimizer(self.c, beta1=config.beta).minimize(self.d_loss, var_list=self.d_vars)
		g_opt = tf.train.AdamOptimizer(self.c, beta1=config.beta).minimize(self.g_loss, var_list=self.g_vars)

		tf.global_variables_initializer().run()

		self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

		sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
		
		if config.dbname == 'mnist':
			sample_inputs = self.data_X[0:self.sample_num]
			sample_labels = self.data_y[0:self.sample_num]
		else:
			sample_files = self.data[0:self.sample_num]
			sample = [get_image(sample_file,input_h=self.in_dim, input_w=self.in_dim, resize_h=self.out_dim,
				resize_w=self.out_dim, crop=self.crop, grayscale=self.grayscale) for sample_file in sample_files]
			if self.grayscale:
				sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
			else:
				sample_inputs = np.array(sample).astype(np.float32)
	
		counter = 1
		start_time = time.time()
		could_load, checkpoint_counter = self.load()
		if could_load:
			counter = checkpoint_counter
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		for epoch in xrange(config.epoch):

			if config.dbname == 'mnist':
				batch_idxs = len(self.data_X) // config.batch_size

			else:      
				self.data = glob(os.path.join("./data", config.dbname, "*.jpg"))
				batch_idxs = len(self.data) // config.batch_size

			for i in xrange(0, batch_idxs):

				if config.dbname == 'mnist':
					batch_images = self.data_X[i*config.batch_size:(i+1)*config.batch_size]
					batch_labels = self.data_y[i*config.batch_size:(i+1)*config.batch_size]

				else:
					batch_files = self.data[i*config.batch_size:(i+1)*config.batch_size]
					batch = [get_image(batch_file, input_h=self.in_dim, input_w=self.in_dim, resize_h=self.out_dim,
						resize_w=self.out_dim, crop=self.crop, grayscale=self.grayscale) for batch_file in batch_files]

					if self.grayscale:
						batch_images = np.array(batch).astype(np.float32)[:, :, :, None]

					else:
						batch_images = np.array(batch).astype(np.float32)

				batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

				if config.dbname == 'mnist':
					self.sess.run([d_opt], feed_dict={self.inputs: batch_images, self.z: batch_z, self.y:batch_labels,})
					self.sess.run([g_opt], feed_dict={self.z: batch_z, self.y:batch_labels,})
					self.sess.run([g_opt], feed_dict={ self.z: batch_z, self.y:batch_labels})
					
					errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.y:batch_labels})
					errD_real = self.d_loss_real.eval({self.inputs: batch_images, self.y:batch_labels})
					errG = self.g_loss.eval({self.z: batch_z, self.y: batch_labels})

				else:
					self.sess.run([d_opt], feed_dict={self.inputs: batch_images, self.z: batch_z})
					self.sess.run([g_opt], feed_dict={self.z: batch_z})
					self.sess.run([g_opt], feed_dict={self.z: batch_z})
					
					errD_fake = self.d_loss_fake.eval({self.z: batch_z})
					errD_real = self.d_loss_real.eval({self.inputs: batch_images})
					errG = self.g_loss.eval({self.z: batch_z})

				counter += 1
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, i, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))

				if np.mod(counter, 100) == 1:
					if config.dbname == 'mnist':
						samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss], feed_dict={self.z: sample_z, self.inputs: sample_inputs, self.y:sample_labels,})
						save_images(samples, image_manifold_size(samples.shape[0]), './{}/train_{:02d}_{:04d}.png'.format(self.smp_dir, epoch, i))
						print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
					else:
						samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss], feed_dict={self.z: sample_z, self.inputs: sample_inputs,},)
						save_images(samples, image_manifold_size(samples.shape[0]), './{}/train_{:02d}_{:04d}.png'.format(self.smp_dir, epoch, i))
						print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

				if np.mod(counter, 500) == 2:
					self.save(config.chk_dir, counter)

	def discriminator(self, image, y=None, reuse=False):

		with tf.variable_scope("discriminator") as scope:

			if reuse:
				scope.reuse_variables()

			if not self.y_dim:

				h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
				h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
				h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
				h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
				h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

				return tf.nn.sigmoid(h4), h4

			else:
				yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
				x = conv_cond_concat(image, yb)

				h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
				h0 = conv_cond_concat(h0, yb)

				h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
				h1 = tf.reshape(h1, [self.batch_size, -1])      
				h1 = tf.concat([h1, y], 1)
				
				h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
				h2 = tf.concat([h2, y], 1)

				h3 = linear(h2, 1, 'd_h3_lin')
				
				return tf.nn.sigmoid(h3), h3


	def generator(self, z, y=None):

		get_shape = lambda x,y : int(math.ceil(float(x)/float(y)))

		with tf.variable_scope("generator") as scope:

			if not self.y_dim:

				s_h,   s_w   = self.out_dim,       self.out_dim
				s_h2,  s_w2  = get_shape(s_h, 2),  get_shape(s_w, 2)
				s_h4,  s_w4  = get_shape(s_h2, 2), get_shape(s_w2, 2)
				s_h8,  s_w8  = get_shape(s_h4, 2), get_shape(s_w4, 2)
				s_h16, s_w16 = get_shape(s_h8, 2), get_shape(s_w8, 2)

				self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

				self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
				h0 = tf.nn.relu(self.g_bn0(self.h0))

				self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
				h1 = tf.nn.relu(self.g_bn1(self.h1))

				h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
				h2 = tf.nn.relu(self.g_bn2(h2))

				h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
				h3 = tf.nn.relu(self.g_bn3(h3))

				h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

				return tf.nn.tanh(h4)

			else:

				s_h,  s_w  = self.out_dim, self.out_dim
				s_h2, s_h4 = int(s_h/2),   int(s_h/4)
				s_w2, s_w4 = int(s_w/2),   int(s_w/4)

				yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
				z  = tf.concat([z, y], 1)

				h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
				h0 = tf.concat([h0, y], 1)

				h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
				h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

				h1 = conv_cond_concat(h1, yb)

				h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
				h2 = conv_cond_concat(h2, yb)

				return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

	def sampler(self, z, y=None):

		get_shape = lambda x,y : int(math.ceil(float(x)/float(y)))

		with tf.variable_scope("generator") as scope:

			scope.reuse_variables()

			if not self.y_dim:

				s_h,   s_w   = self.out_dim,       self.out_dim
				s_h2,  s_w2  = get_shape(s_h, 2),  get_shape(s_w, 2)
				s_h4,  s_w4  = get_shape(s_h2, 2), get_shape(s_w2, 2)
				s_h8,  s_w8  = get_shape(s_h4, 2), get_shape(s_w4, 2)
				s_h16, s_w16 = get_shape(s_h8, 2), get_shape(s_w8, 2)

				h0 = tf.reshape(linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'), [-1, s_h16, s_w16, self.gf_dim * 8])
				h0 = tf.nn.relu(self.g_bn0(h0, train=False))

				h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
				h1 = tf.nn.relu(self.g_bn1(h1, train=False))

				h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
				h2 = tf.nn.relu(self.g_bn2(h2, train=False))

				h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
				h3 = tf.nn.relu(self.g_bn3(h3, train=False))

				h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

				return tf.nn.tanh(h4)

			else:
				s_h,  s_w  = self.out_dim, self.out_dim
				s_h2, s_h4 = int(s_h/2),   int(s_h/4)
				s_w2, s_w4 = int(s_w/2),   int(s_w/4)

				yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
				z = tf.concat([z, y], 1)

				h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
				h0 = tf.concat([h0, y], 1)

				h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
				h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
				h1 = conv_cond_concat(h1, yb)

				h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
				h2 = conv_cond_concat(h2, yb)

				return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))


	def load_mnist(self):

		data_dir = os.path.join("./data", self.dbname)
		
		fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
		loaded = np.fromfile(file=fd,dtype=np.uint8)
		trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

		fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
		loaded = np.fromfile(file=fd,dtype=np.uint8)
		trY = loaded[8:].reshape((60000)).astype(np.float)

		fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
		loaded = np.fromfile(file=fd,dtype=np.uint8)
		teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

		fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
		loaded = np.fromfile(file=fd,dtype=np.uint8)
		teY = loaded[8:].reshape((10000)).astype(np.float)

		trY = np.asarray(trY)
		teY = np.asarray(teY)
		
		X = np.concatenate((trX, teX), axis=0)
		y = np.concatenate((trY, teY), axis=0).astype(np.int)
		
		seed = 547
		np.random.seed(seed)
		np.random.shuffle(X)
		np.random.seed(seed)
		np.random.shuffle(y)
		
		y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)

		for i, label in enumerate(y):
			y_vec[i,y[i]] = 1.0
		
		return X/255.,y_vec


	@property
	def model_dir(self):
		return "{}_{}_{}_{}".format(self.dbname, self.batch_size, self.out_dim, self.out_dim)


	def save(self, chk_dir, step):

		model_name = "DCGAN.model"
		chk_dir = os.path.join(chk_dir, self.model_dir)
		check_dirs([chk_dir])
		self.saver.save(self.sess, os.path.join(chk_dir, model_name), global_step=step)


	def load(self):

		import re
		print(" [*] Reading checkpoints...")
		chk_dir = os.path.join(self.chk_dir, self.model_dir)

		ckpt = tf.train.get_checkpoint_state(chk_dir)

		if ckpt and ckpt.model_checkpoint_path:

			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(chk_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter

		else:

			print(" [*] Failed to find a checkpoint")
			return False, 0
