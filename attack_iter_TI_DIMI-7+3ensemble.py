# -*- coding : utf-8 -*-
# coding: utf-8
"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

start_time = time.time()

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
import csv
import tensorflow as tf

from nets import inception_v3, inception_v4, inception_resnet_v2,resnet_v2,densenet
from nets.nasnet import pnasnet
from nets.core import xception

# from nets.mobilenet import mobilenet_v2
# from nets.nasnet import nasnet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU selection"0,1" means choose 0# and 1# GPU
slim = tf.contrib.slim

tf.flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string('checkpoint_path_inception_v3',
                       './models/inception_v3/inception_v3.ckpt',
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_adv_inception_v3',
                        './models/adv_inception_v3/adv_inception_v3_rename.ckpt',
                        'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_inception_v4',
                       './models/inception_v4/inception_v4.ckpt',
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_inception_resnet_v2',
                       './models/inception_resnet_v2_2016_08_30/inception_resnet_v2_2016_08_30.ckpt',
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_resnet_v2_152',
                       './models/resnet_v2_152/resnet_v2_152.ckpt',
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_ens_adv_inception_resnet_v2',
                       './models/ens_adv_inception_resnet_v2/ens_adv_inception_resnet_v2_rename.ckpt',
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_ens3_adv_inception_v3',
    './models/ens3_adv_inception_v3/ens3_adv_inception_v3_rename.ckpt',
    'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_ens4_adv_inception_v3',
    './models/ens4_adv_inception_v3/ens4_adv_inception_v3_rename.ckpt',
    'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_densenet169', './models/tf-densenet169/tf-densenet169.ckpt', 'Path to checkpoint for resnet network.')

tf.flags.DEFINE_string(
    'checkpoint_path_pnasnet', './models/pnasnet-5_large_2017_12_13/model.ckpt', 'Path to checkpoint for pnasnet network.')

tf.flags.DEFINE_string(
    'checkpoint_path_xception_71', './models/xception_71_2018_05_09/model.ckpt', 'Path to checkpoint for xception network.')



tf.flags.DEFINE_string('input_dir', './images', 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './TI-DI-MI/out_7+3', 'Output directory with images.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_iter', 10, 'umber of iterations.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer('image_resize', 330, 'Height of each input images.')

tf.flags.DEFINE_integer('batch_size', 2, 'How many images process at one time.')#more models,less batch_size

tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_float('prob', 0, 'probability of using diverse inputs.')

FLAGS = tf.flags.FLAGS


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


kernel2 = gkern(7, 3).astype(np.float32)
stack_kernel2 = np.stack([kernel2, kernel2, kernel2]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel2, 3)


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    labels = np.zeros(batch_shape[0], dtype=np.int32)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    with open('./dev_dataset.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = os.path.join(input_dir, row['ImageId'] + '.png')
            try:
                with open(filepath, 'rb') as f:
                    raw_image = imread(f, mode='RGB').astype(np.float)
                    image = imresize(raw_image, [FLAGS.image_height, FLAGS.image_width]) / 255.0
            except:
                continue
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            images[idx, :, :, :] = image * 2.0 - 1.0
            labels[idx] = int(row['TrueLabel'])
            filenames.append(os.path.basename(filepath))
            idx += 1
            if idx == batch_size:
                yield filenames, images, labels
                filenames = []
                images = np.zeros(batch_shape)
                labels = np.zeros(batch_shape[0], dtype=np.int32)
                idx = 0
        if idx > 0:
            yield filenames, images, labels


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def graph3(x, y, i, x_max, x_min, grad):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    momentum = FLAGS.momentum
    alpha = eps / num_iter
    num_classes = 1001

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            input_diversity(x), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_v4, end_points_v4 = inception_v4.inception_v4(
            input_diversity(x), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
            input_diversity(x), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet, end_points_resnet = resnet_v2.resnet_v2_101(
            input_diversity(x), num_classes=num_classes, is_training=False, scope='resnet_v2_101', reuse=tf.AUTO_REUSE)

    with slim.arg_scope(densenet.densenet_arg_scope()):
        logits_densenet169, end_points_densenet169 = densenet.densenet169(
            input_diversity(x), num_classes=1000, is_training=False,reuse=tf.AUTO_REUSE)
    logits_densenet169 = tf.pad((logits_densenet169), [[0, 0], [1, 0]])

    with slim.arg_scope(xception.xception_arg_scope()):
        logits_xception_71, end_points_xception_71 = xception.xception_71(
            input_diversity(x), num_classes=num_classes, is_training=False,reuse=tf.AUTO_REUSE)

    with slim.arg_scope(pnasnet.pnasnet_large_arg_scope()):
        logits_pnasnet, end_points_pnasnet = pnasnet.build_pnasnet_large(
            input_diversity(x), num_classes=1001, is_training=False)

    #ensemble
    logits = (logits_v3 + logits_v4 + logits_res_v2 + logits_resnet + logits_densenet169 + logits_xception_71 + logits_pnasnet) / 7
    auxlogits = (end_points_v3['AuxLogits'] + end_points_v4['AuxLogits'] + end_points_res_v2['AuxLogits'] ) / 3

    cross_entropy = tf.losses.softmax_cross_entropy(y,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(y,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=0.4)

    noise = tf.gradients(cross_entropy, x)[0]
    noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
    noise = momentum * grad + noise
    noise1 = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    x = x + alpha * tf.clip_by_value(noise1,-1,1)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise


def stop(x, y, i, x_max, x_min, grad):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)


def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    rescaled_input = tf.image.resize_images(padded, [299, 299], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    rescaled_input.set_shape((input_tensor.shape[0], 299, 299, 3))
    return rescaled_input


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_classes = 1001
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    print(time.time() - start_time)

    with tf.Graph().as_default():
        # Prepare graph

        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)
        target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        y = tf.one_hot(target_class_input, num_classes)
        i = tf.constant(0, dtype=tf.float32)
        grad = tf.zeros(shape=batch_shape)

        x_adv10, _, _, _, _, _ = tf.while_loop(stop, graph3, [x_input, y, i, x_max, x_min, grad])

        nb_classes = 1001

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
                x_adv10, num_classes=nb_classes, is_training=False, scope='Ens3AdvInceptionV3', reuse=tf.AUTO_REUSE)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
                x_adv10, num_classes=nb_classes, is_training=False, scope='Ens4AdvInceptionV3', reuse=tf.AUTO_REUSE)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
                x_adv10, num_classes=nb_classes, is_training=False, scope='EnsAdvInceptionResnetV2', reuse=tf.AUTO_REUSE)

        #ensemble the attacked defense-models
        logits = (logits_ens3_adv_v3 + logits_ens4_adv_v3 + logits_res_v2) / 3

        correct_prediction1 = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, 'float'))

        # Run computation
        #note that the difference between ensembled modles and single model

        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_101'))

        s5 = tf.train.Saver(slim.get_model_variables(scope='densenet169'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='xception_71'))
        s7 = tf.train.Saver(slim.get_model_variables()[4026:5205])

        # s8 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))

        s9 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        s10 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s11 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))


        with tf.Session() as sess:
            s1.restore(sess, FLAGS.checkpoint_path_inception_v3)
            s2.restore(sess, FLAGS.checkpoint_path_inception_v4)
            s3.restore(sess, FLAGS.checkpoint_path_inception_resnet_v2)
            s4.restore(sess, FLAGS.checkpoint_path_resnet)
            s5.restore(sess, FLAGS.checkpoint_path_densenet169)
            s6.restore(sess, FLAGS.checkpoint_path_xception_71)
            s7.restore(sess, FLAGS.checkpoint_path_pnasnet)
            # s8.restore(sess, FLAGS.checkpoint_path_adv_inception_v3)

            s9.restore(sess, FLAGS.checkpoint_path_ens3_adv_inception_v3)
            s10.restore(sess, FLAGS.checkpoint_path_ens4_adv_inception_v3)
            s11.restore(sess, FLAGS.checkpoint_path_ens_adv_inception_resnet_v2)



            print(time.time() - start_time)
            for k in range(80, 81):

                for filenames, images, tlabels in load_images(FLAGS.input_dir, batch_shape):
                    adv_images, a1 = sess.run(
                        [x_adv10, accuracy1],
                         feed_dict={x_input: images, target_class_input: tlabels})

                    save_images(adv_images, filenames, FLAGS.output_dir)
                    accuracy_set1.append(a1)

                print(k)
                mean_acc1 = np.mean(accuracy_set1)
                print(1 - mean_acc1)

                print('-----------------------------------------------')
            print(time.time() - start_time)



if __name__ == '__main__':
    tf.app.run()

# -*- encoding: utf-8 -*-