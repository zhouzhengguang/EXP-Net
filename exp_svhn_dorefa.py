#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: svhn-digit-dorefa.py
# Author: Yuxin Wu

import argparse
import os
import numpy as np
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils.varreplace import remap_variables
import tensorflow as tf

from dorefa import get_dorefa
from dorefa import ternarize

"""
This is a tensorpack script for the SVHN results in paper:
DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
http://arxiv.org/abs/1606.06160
The original experiements are performed on a proprietary framework.
This is our attempt to reproduce it on tensorpack/tensorflow.
Accuracy:
    With (W,A,G)=(1,1,4), can reach 3.1~3.2% error after 150 epochs.
    With the GaussianDeform augmentor, it will reach 2.8~2.9%
    (we are not using this augmentor in the paper).
    With (W,A,G)=(1,2,4), error is 3.0~3.1%.
    With (W,A,G)=(32,32,32), error is about 2.9%.
Speed:
    30~35 iteration/s on 1 TitanX Pascal. (4721 iterations / epoch)
To Run:
    ./svhn-digit-dorefa.py --dorefa 1,2,4
"""

BITW = 1
BITA = 2
BITG = 32


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 40, 40, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        is_training = get_current_tower_context().is_training

        fw, fa, fg = get_dorefa(BITW, BITA, BITG)

        # monkey-patch tf.get_variable to apply fw
        def binarize_weight(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'conv0' in name or 'weak' in name or 'fc' in name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)
                #return ternarize(v)

        def cabs(x):
            return tf.minimum(1.0, tf.abs(x), name='cabs')

        def activate(x):
            return fa(cabs(x))
        
        def merge(x, y):
            #return x + y
            #return x - y
            return tf.concat([x,y], axis=3)

        image = image / 256.0;          k=3;      zp=0.25;      zp2=zp / 1
        #scale = tf.train.exponential_decay(learning_rate=1.0, global_step=get_global_step_var(), decay_steps=4721*5, decay_rate=0.5, staircase=True, name='scale')
        #scale = tf.where(scale>0.001, scale, tf.zeros_like(scale))
        scale = tf.train.cosine_decay(learning_rate=1.0, global_step=get_global_step_var(), decay_steps=4721*50, alpha=0.0)
        tf.summary.scalar('scale', scale);             endconv=[];  endweak=[]
        #scale2 = tf.train.cosine_decay(learning_rate=1.0, global_step=get_global_step_var(), decay_steps=4721*50, alpha=0.0)
        #scale3 = tf.train.cosine_decay(learning_rate=1.0, global_step=get_global_step_var(), decay_steps=4721*80, alpha=0.0)
        with remap_variables(binarize_weight), \
                argscope(BatchNorm, momentum=0.9, epsilon=1e-4), \
                argscope(Conv2D, use_bias=False):
            net=Conv2D('conv0', image, np.round(48*zp), 5, padding='VALID', use_bias=True)
            net=MaxPooling('pool0', net, 2, padding='SAME');            net=activate(net)

            net1=Conv2D('conv1', net, np.round(64*zp), 3, padding='SAME');      net1=BatchNorm('bn1', net1);     endconv.append(net1)
            net2=Conv2D('weak1', net, np.round(64*zp2), k, padding='SAME');      net2=BatchNorm('bn12', net2);      endweak.append(net2);  #  net2=tf.nn.relu(net2)
            net=merge(activate(net1), scale*net2)
            #net=activate(net1)
            
            net1=Conv2D('conv2', net, np.round(64*zp), 3, padding='SAME');      net1=BatchNorm('bn2', net1);     endconv.append(net1)
            net2=Conv2D('weak2', net, np.round(64*zp2), k, padding='SAME');      net2=BatchNorm('bn22', net2);      endweak.append(net2);   # net2=tf.nn.relu(net2)
            net1=MaxPooling('pool1', net1, 2, padding='SAME');   net2=MaxPooling('pool12', net2, 2, padding='SAME');
            net=merge(activate(net1), scale*net2)
            net=activate(net1)

            net1=Conv2D('conv3', net, np.round(128*zp), 3, padding='VALID');      net1=BatchNorm('bn3', net1);     endconv.append(net1)
            net2=Conv2D('weak3', net, np.round(128*zp2), k, padding='VALID');      net2=BatchNorm('bn32', net2);      endweak.append(net2);  #  net2=tf.nn.relu(net2)
            net=merge(activate(net1), scale*net2)
            #net=activate(net1)

            net1=Conv2D('conv4', net, np.round(128*zp), 3, padding='SAME');      net1=BatchNorm('bn4', net1);     endconv.append(net1)
            net2=Conv2D('weak4', net, np.round(128*zp2), k, padding='SAME');      net2=BatchNorm('bn42', net2);      endweak.append(net2);  #  net2=tf.nn.relu(net2)
            net=merge(activate(net1), scale*net2)
            # net=activate(net1)

            net1=Conv2D('conv5', net, np.round(128*zp), 3, padding='VALID');      net1=BatchNorm('bn5', net1);     endconv.append(net1)
            net2=Conv2D('weak5', net, np.round(128*zp2), k, padding='VALID');      net2=BatchNorm('bn52', net2);      endweak.append(net2);  #  net2=tf.nn.relu(net2)
            net=merge(activate(net1), scale*net2)
            #net=activate(net1)

            net=tf.nn.dropout(net, 0.5 if is_training else 1.0)
            net1=Conv2D('conv6', net, np.round(512*zp), 5, padding='VALID');       net1=BatchNorm('bn6', net1);     endconv.append(net1)
            net2=Conv2D('weak6', net, np.round(512*zp2), 5, padding='VALID');       net2=BatchNorm('bn62', net2);      endweak.append(net2);  #  net2=tf.nn.relu(net2)
            net=merge(cabs(net1), scale*net2)
            # net=cabs(net1)
            logits=FullyConnected('fc1', net, 10)
        tf.nn.softmax(logits, name='output')

        # compute the number of failed samples
        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='wrong_tensor')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(1e-7))

        add_param_summary(('.*/W', ['histogram', 'rms']))
        total_cost = tf.add_n([cost, wd_cost], name='cost')
        add_moving_summary(cost, wd_cost, total_cost)
        for i in range(len(endweak)):
            add_moving_summary(tf.reduce_mean(tf.abs(endconv[i]), name='mean_conv_'+str(i+1) )  )
            add_moving_summary(tf.reduce_mean(tf.abs(endweak[i]), name='mean_weak_'+str(i+1) )  )

        return total_cost

    def optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=4721 * 100,
            decay_rate=0.5, staircase=True, name='learning_rate')
        #lr=tf.get_variable('learning_rate', initializer=2e-4, trainable=False)
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr, epsilon=1e-5)


def get_config():
    #logger.auto_set_dir()
    logger.set_logger_dir('./train_log/fp_cat_z1_svhn_dorefa_cos50_zp_025_'+str(args.dorefa)+'_exp1')
    # prepare dataset
    d1 = dataset.SVHNDigit('train')
    d2 = dataset.SVHNDigit('extra')
    data_train = RandomMixData([d1, d2])
    data_test = dataset.SVHNDigit('test')

    augmentors = [
        imgaug.Resize((40, 40)),
        imgaug.Brightness(30),
        imgaug.Contrast((0.5, 1.5)),
        # imgaug.GaussianDeform(  # this is slow but helpful. only use it when you have lots of cpus
        # [(0.2, 0.2), (0.2, 0.8), (0.8,0.8), (0.8,0.2)],
        # (40,40), 0.2, 3),
    ]
    data_train = AugmentImageComponent(data_train, augmentors)
    data_train = BatchData(data_train, 128)
    data_train = PrefetchDataZMQ(data_train, 5)

    augmentors = [imgaug.Resize((40, 40))]
    data_test = AugmentImageComponent(data_test, augmentors)
    data_test = BatchData(data_test, 128, remainder=True)

    return TrainConfig(
        data=QueueInput(data_train),
        callbacks=[
            ModelSaver(max_to_keep=2),
            # ScheduledHyperParamSetter(
                 # 'learning_rate', [(1, 1e-4), (2, 2e-5), (4, 3e-6)]),
            InferenceRunner(data_test,
                            [ScalarStats('cost'), ClassificationError('wrong_tensor')])
        ],
        model=Model(),
        max_epoch=200,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--dorefa',
                        help='number of bits for W,A,G, separated by comma. Defaults to \'1,2,4\'',
                        default='1,2,4')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    BITW, BITA, BITG = map(int, args.dorefa.split(','))
    config = get_config()
    #launch_train_with_config(config, SimpleTrainer())
    nr_gpu = max(get_nr_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpu))
