import argparse
import numpy as np
import tensorflow as tf

import vgg19
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, help='Filename of image')
args = parser.parse_args()

img = utils.load_image(args.img or './test-data/tiger.jpeg')
batch = img.reshape((1, 224, 224, 3))

with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [1, 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg19.Vgg19()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        utils.print_prob(prob[0], './synset.txt')
