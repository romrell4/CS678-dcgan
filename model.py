from __future__ import division

import time
from glob import glob

from ops import *
from utils import *

class DCGAN(object):
    def __init__(self, sess, config):
        self.sess = sess

        # Save the config settings as instance variables
        self.c = config.c
        self.beta = config.beta
        self.crop = config.crop
        self.epoch = config.epoch
        self.dbname = config.dbname
        self.chk_dir = config.chk_dir
        self.smp_dir = config.smp_dir
        self.z_dim = config.test_size
        self.training = config.training
        self.batch_size = config.batch_size
        self.sample_num = config.batch_size

        # TODO: Figure out what this variable is for
        self.gf_dim = self.df_dim = 64
        # Setup the dimensions for the generator/discriminator
        self.gfc_dim = self.dfc_dim = 1024

        # Create the batch normalization layers ahead of time
        self.d_bn1 = BatchNorm(name = 'd_bn1')
        self.d_bn2 = BatchNorm(name = 'd_bn2')
        self.g_bn0 = BatchNorm(name = 'g_bn0')
        self.g_bn1 = BatchNorm(name = 'g_bn1')
        self.g_bn2 = BatchNorm(name = 'g_bn2')

        # Load the data based on which dataset was requested
        if self.dbname == 'mnist':
            self.in_dim, self.out_dim, self.y_dim = 28, 28, 10
            self.data_x, self.data_y = self.load_mnist()
            self.c_dim = self.data_x[0].shape[-1]

        else:
            self.in_dim, self.out_dim, self.y_dim = 108, 64, None
            self.data = glob(os.path.join("./data", self.dbname, "*.jpg"))
            shape = imread(self.data[0]).shape
            self.c_dim = 1 if len(shape) < 3 else shape[-1]
            self.d_bn3 = BatchNorm(name = 'd_bn3')
            self.g_bn3 = BatchNorm(name = 'g_bn3')

        self.grayscale = (self.c_dim == 1)

        self.build_model()

        if self.training:
            self.train()
        else:
            if not self.load():
                raise Exception("[!] Train a model first, then run test mode")

    def build_model(self):

        if self.dbname == "mnist":
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name = 'y')
        else:
            self.y = None

        if self.crop:
            image_dims = [self.out_dim, self.out_dim, self.c_dim]
        else:
            image_dims = [self.in_dim, self.in_dim, self.c_dim]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name = 'real_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name = 'z')

        self.G = self.generator(self.z, self.y)
        self.D, self.D_logits = self.discriminator(self.inputs, self.y)
        self.sampler = self.generator(self.z, self.y, reuse = True)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse = True)

        # TODO: Understand this
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.D), logits = self.D_logits))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(self.D_), logits = self.D_logits_))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.D_), logits = self.D_logits_))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self):

        d_opt = tf.train.AdamOptimizer(self.c, beta1 = self.beta).minimize(self.d_loss, var_list = self.d_vars)
        g_opt = tf.train.AdamOptimizer(self.c, beta1 = self.beta).minimize(self.g_loss, var_list = self.g_vars)

        tf.global_variables_initializer().run()

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size = (self.sample_num, self.z_dim))

        if self.dbname == 'mnist':
            sample_inputs = self.data_x[0:self.sample_num]
            sample_labels = self.data_y[0:self.sample_num]
        else:
            sample_files = self.data[0:self.sample_num]
            sample = [get_image(sample_file, input_h = self.in_dim, input_w = self.in_dim, resize_h = self.out_dim,
                                resize_w = self.out_dim, crop = self.crop, grayscale = self.grayscale) for sample_file in sample_files]
            if self.grayscale:
                sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_inputs = np.array(sample).astype(np.float32)

        start_time = time.time()
        counter = self.load()

        for epoch in xrange(self.epoch):

            if self.dbname == 'mnist':
                batch_idxs = len(self.data_x) // self.batch_size

            else:
                # TODO: Is the duplicate code? Didn't we already do this above?
                self.data = glob(os.path.join("./data", self.dbname, "*.jpg"))
                batch_idxs = len(self.data) // self.batch_size

            for i in xrange(0, batch_idxs):

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                if self.dbname == 'mnist':
                    batch_images = self.data_x[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_labels = self.data_y[i * self.batch_size:(i + 1) * self.batch_size]

                else:
                    batch_files = self.data[i * self.batch_size:(i + 1) * self.batch_size]
                    batch = [get_image(batch_file, input_h = self.in_dim, input_w = self.in_dim, resize_h = self.out_dim,
                                       resize_w = self.out_dim, crop = self.crop, grayscale = self.grayscale) for batch_file in batch_files]

                    if self.grayscale:
                        batch_images = np.array(batch).astype(np.float32)[:, :, :, None]

                    else:
                        batch_images = np.array(batch).astype(np.float32)

                # if self.dbname == 'mnist':
                #     self.sess.run([d_opt], feed_dict = {self.inputs: batch_images, self.z: batch_z, self.y: batch_labels, })
                #     self.sess.run([g_opt], feed_dict = {self.z: batch_z, self.y: batch_labels, })
                #     self.sess.run([g_opt], feed_dict = {self.z: batch_z, self.y: batch_labels})

                #     errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.y: batch_labels})
                #     errD_real = self.d_loss_real.eval({self.inputs: batch_images, self.y: batch_labels})
                #     errG = self.g_loss.eval({self.z: batch_z, self.y: batch_labels})

                # else:
                #     self.sess.run([d_opt], feed_dict = {self.inputs: batch_images, self.z: batch_z})
                #     self.sess.run([g_opt], feed_dict = {self.z: batch_z})
                #     self.sess.run([g_opt], feed_dict = {self.z: batch_z})

                #     errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                #     errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                #     errG = self.g_loss.eval({self.z: batch_z})

                d_dict = {self.inputs: batch_images, self.z: batch_z}
                g_dict = {self.z: batch_z}
                x_dict = {self.inputs: batch_images}

                if self.dbname == 'mnist':
                    d_dict[self.y] = batch_labels
                    g_dict[self.y] = batch_labels
                    x_dict[self.y] = batch_labels

                self.sess.run([d_opt], feed_dict = d_dict)
                self.sess.run([g_opt], feed_dict = g_dict)
                self.sess.run([g_opt], feed_dict = g_dict)

                errD_fake = self.d_loss_fake.eval(g_dict)
                errD_real = self.d_loss_real.eval(x_dict)
                errG = self.g_loss.eval(g_dict)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, i, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 100) == 1:

                    d = {self.z: sample_z, self.inputs: sample_inputs}

                    if self.dbname == 'mnist':
                        d[self.y] = sample_labels

                    samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss], feed_dict=d)
                    save_images(samples, image_manifold_size(samples.shape[0]), './{}/train_{:02d}_{:04d}.png'.format(self.smp_dir, epoch, i))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(self.chk_dir, counter)

    def discriminator(self, image, y = None, reuse = False):

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            if self.dbname == "mnist":
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = conv_cond_concat(lrelu(conv2d(x, self.c_dim + self.y_dim, name = 'd_h0_conv')), yb)
                h1 = tf.concat([tf.reshape(lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name = 'd_h1_conv'))), [self.batch_size, -1]), y], 1)
                h2 = tf.concat([lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin'))), y], 1)
                h3 = linear(h2, 1, 'd_h3_lin')
                return tf.nn.sigmoid(h3), h3
            else:
                h0 = lrelu(conv2d(image, self.df_dim, name = 'd_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name = 'd_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name = 'd_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name = 'd_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
                return tf.nn.sigmoid(h4), h4

    def generator(self, z, y = None, reuse = False):

        def get_shape(x, y):
            return int(math.ceil(float(x) / float(y)))

        with tf.variable_scope("generator") as scope:

            if reuse:
                scope.reuse_variables()

            train = not reuse

            if self.dbname == "mnist":
                # TODO: Use the lambda above to round up instead of cast to int which rounds down
                s_dim = self.out_dim
                s_dim2 = int(s_dim / 2)
                s_dim4 = int(s_dim / 4)

                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat([z, y], 1)

                h0 = tf.concat([tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train = train)), y], 1)
                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim * 2 * s_dim4 * s_dim4, 'g_h1_lin'), train = train))
                h1 = conv_cond_concat(tf.reshape(h1, [self.batch_size, s_dim4, s_dim4, self.gf_dim * 2]), yb)
                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_dim2, s_dim2, self.gf_dim * 2], name = 'g_h2'), train = train))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_dim, s_dim, self.c_dim], name = 'g_h3'))
            else:
                s_dim = self.out_dim
                s_dim2 = get_shape(s_dim, 2)
                s_dim4 = get_shape(s_dim2, 2)
                s_dim8 = get_shape(s_dim4, 2)
                s_dim16 = get_shape(s_dim8, 2)

                z_ = linear(z, self.gf_dim * 8 * s_dim16 * s_dim16, 'g_h0_lin')

                h0 = tf.nn.relu(self.g_bn0(tf.reshape(z_, [-1, s_dim16, s_dim16, self.gf_dim * 8]), train = train))
                h1 = tf.nn.relu(self.g_bn1(deconv2d(h0, [self.batch_size, s_dim8, s_dim8, self.gf_dim * 4], name = 'g_h1'), train = train))
                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_dim4, s_dim4, self.gf_dim * 2], name = 'g_h2'), train = train))
                h3 = tf.nn.relu(self.g_bn3(deconv2d(h2, [self.batch_size, s_dim2, s_dim2, self.gf_dim * 1], name = 'g_h3'), train = train))
                return tf.nn.tanh(deconv2d(h3, [self.batch_size, s_dim, s_dim, self.c_dim], name = 'g_h4'))

    def load_mnist(self):

        data_dir = os.path.join("./data", self.dbname)

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file = fd, dtype = np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file = fd, dtype = np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file = fd, dtype = np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file = fd, dtype = np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis = 0)
        y = np.concatenate((trY, teY), axis = 0).astype(np.int)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_vec = np.zeros((len(y), self.y_dim), dtype = np.float)

        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        return X / 255., y_vec

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(self.dbname, self.batch_size, self.out_dim, self.out_dim)

    def save(self, chk_dir, step):

        model_name = "DCGAN.model"
        chk_dir = os.path.join(chk_dir, self.model_dir)
        check_dirs([chk_dir])
        self.saver.save(self.sess, os.path.join(chk_dir, model_name), global_step = step)

    def load(self):

        import re

        chk_dir = os.path.join(self.chk_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(chk_dir)

        if ckpt and ckpt.model_checkpoint_path:

            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(chk_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Loaded {} successfully".format(ckpt_name))
            return counter

        else:

            print(" [*] No checkpoint found")
            return 1
