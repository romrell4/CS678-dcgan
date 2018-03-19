import tensorflow as tf

from model import DCGAN
from utils import visualize, check_dirs


f = tf.app.flags
f.DEFINE_integer("epoch",		25,				"Number of epochs [25]")
f.DEFINE_float  ("c",			0.0002,			"Learning rate [0.0002]")
f.DEFINE_float  ("beta",		0.5,			"Momentum [0.5]")
f.DEFINE_integer("batch_size",	64,				"Number of images in batch [64]")
f.DEFINE_string ("dbname",		"celebA",		"Name of dataset [celebA, mnist, lsun]")
f.DEFINE_string ("chk_dir",		"checkpoint",	"Directory name to save the checkpoints [checkpoint]")
f.DEFINE_string ("smp_dir",		"samples",		"Directory name to save the image samples [samples]")
f.DEFINE_boolean("training",	False,			"True for training, False for testing [False]")
f.DEFINE_boolean("crop",		False,			"True for training, False for testing [False]")
f.DEFINE_boolean("visualize",	False,			"True for visualizing, False for nothing [False]")
f.DEFINE_integer("test_size",	100,			"Number of images to generate during test. [100]")
config = f.FLAGS

def main(_):

	check_dirs([config.chk_dir, config.smp_dir])

	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

	with tf.Session(config=run_config) as sess:
		dcgan = DCGAN(sess, config)
		visualize(sess, dcgan, config, option=1)

if __name__ == '__main__':
	tf.app.run()
