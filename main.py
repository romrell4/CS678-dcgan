import tensorflow as tf

from model import DCGAN
from utils import visualize, check_dirs

# python main.py --dbname mnist --training
# python main.py --dbname celebA --training --crop

# This is tensorflow wrapper around argparse, and is just allowing you to set the parameters from the command line
f = tf.app.flags
f.DEFINE_integer("epoch", 25, "Number of epochs [25]")
f.DEFINE_float("c", 0.0002, "Learning rate [0.0002]")
f.DEFINE_float("beta", 0.5, "Momentum [0.5]")
f.DEFINE_integer("batch_size", 64, "Number of images in batch [64]")
f.DEFINE_string("dbname", "celebA", "Name of dataset [celebA, mnist, lsun]")
f.DEFINE_string("chk_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
f.DEFINE_string("smp_dir", "samples", "Directory name to save the image samples [samples]")
f.DEFINE_boolean("training", False, "True for training, False for testing [False]")
f.DEFINE_boolean("crop", False, "True for training, False for testing [False]") # TODO: Can we remove this parameter, since it's the same as "training"?
f.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
f.DEFINE_integer("test_size", 100, "Number of images to generate during test. [100]")
config = f.FLAGS

def main(_):
    # Make sure that the directories to save data have been created
    check_dirs([config.chk_dir, config.smp_dir])

    # Set up tensorflow to only use the GPU resources it needs, and to grow when more is necessary
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config = run_config) as sess:
        # Create and train the GAN, then visualize the results
        dcgan = DCGAN(sess, config)
        visualize(sess, dcgan, config, option = 1)

if __name__ == '__main__':
    tf.app.run()
