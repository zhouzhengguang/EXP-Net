# -*- coding: utf-8 -*-
# File: dorefa.py
# Author: Yuxin Wu

import tensorflow as tf
from tensorpack.utils.argtools import graph_memoized

G = tf.get_default_graph()
@graph_memoized
def get_dorefa(bitW, bitA, bitG):
    """
    return the three quantization functions fw, fa, fg, for weights, activations and gradients respectively
    It's unsafe to call this function multiple times with different parameters
    """
    def quantize(x, k):
        n = float(2 ** k - 1)

        @tf.custom_gradient
        def _quantize(x):
            return tf.round(x * n) / n, lambda dy: dy

        return _quantize(x)

    def fw(x):
        if bitW == 32:
            return x

        if bitW == 1:   # BWN
            E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))     #  dorefa-net
            #E = tf.stop_gradient(tf.reduce_mean(tf.abs(x),axis=[0,1,2]))    #   xnor-net

            @tf.custom_gradient
            def _sign(x):
                return tf.sign(x / E) * E, lambda dy: dy

            return _sign(x)

        x = tf.tanh(x)
        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
        return 2 * quantize(x, bitW) - 1

        # return ternarize(x, thresh=0.05)
        #weights=None
        #return fine_grained_quant(x, 0.05, x.op.name, False, weights)

        
        
    def fa(x):
        if bitA == 32:
            return x
        return quantize(x, bitA)

    def fg(x):
        if bitG == 32:
            return x

        @tf.custom_gradient
        def _identity(input):
            def grad_fg(x):
                rank = x.get_shape().ndims
                assert rank is not None
                maxx = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
                x = x / maxx
                n = float(2**bitG - 1)
                x = x * 0.5 + 0.5 + tf.random_uniform(
                    tf.shape(x), minval=-0.5 / n, maxval=0.5 / n)
                x = tf.clip_by_value(x, 0.0, 1.0)
                x = quantize(x, bitG) - 0.5
                return x * maxx * 2

            return input, grad_fg

        return _identity(x)
    return fw, fa, fg


def ternarize(x, thresh=0.05):
    """
    Implemented Trained Ternary Quantization:
    https://arxiv.org/abs/1612.01064
    Code modified from the authors' at:
    https://github.com/czhu95/ternarynet/blob/master/examples/Ternary-Net/ternary.py
    """
    shape = x.get_shape();  # print('ternarize: ', x.get_shape().as_list())

    thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thresh)

    w_p = tf.get_variable('Wp', initializer=1.0, dtype=tf.float32)
    w_n = tf.get_variable('Wn', initializer=1.0, dtype=tf.float32)

    tf.summary.scalar(w_p.op.name + '-summary', w_p)
    tf.summary.scalar(w_n.op.name + '-summary', w_n)

    mask = tf.ones(shape)
    mask_p = tf.where(x > thre_x, tf.ones(shape) * w_p, mask)
    mask_np = tf.where(x < -thre_x, tf.ones(shape) * w_n, mask_p)
    mask_z = tf.where((x < thre_x) & (x > - thre_x), tf.zeros(shape), mask)

    @tf.custom_gradient
    def _sign_mask(x):
        return tf.sign(x) * mask_z, lambda dy: dy

    w = _sign_mask(x)

    w = w * mask_np

    tf.summary.histogram(w.name, w)
    return w
	
def fine_grained_quant(x, eta, name, INITIAL, value, binary=True):
    
    shape = x.get_shape()

    eta_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * eta)

    list_of_masks = []

    if 'conv' in name:

        if INITIAL:
            w_s = tf.get_variable('Ws', [(shape[0].value*shape[1].value),1], collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'SCALING'], initializer=tf.constant_initializer(value[name]))
        else:
            w_s = tf.get_variable('Ws', [(shape[0].value*shape[1].value),1], collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'SCALING'], initializer=tf.constant_initializer(0.2))
        #scalar summary
        # for i in range(0,(shape[0].value*shape[1].value)):
        #     tf.scalar_summary(w_s.name + str(i) +str(0), w_s[i,0])
        
        #each pixel
        for i in range(shape[0].value):
            for j in range(shape[1].value):
                ws = w_s[(shape[1].value*i) + j, 0]
                mask = tf.ones(shape)
                mask_p = tf.where(x[i,j,:,:] > eta_x, mask[i,j,:,:] * ws, mask[i,j,:,:])
                mask_np = tf.where(x[i,j,:,:] < -eta_x, mask[i,j,:,:] * ws, mask_p)
                list_of_masks.append(mask_np)
                
        masker = tf.stack(list_of_masks)
        masker = tf.reshape(masker, [i.value for i in shape])

        if binary:
            mask_z = tf.ones(shape)
        else:
            mask_z = tf.where((x < eta_x) & (x > - eta_x), tf.zeros(shape), tf.ones(shape))

        with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
            w =  tf.sign(x) * tf.stop_gradient(mask_z)

        w = w * masker

        tf.summary.histogram(w.name, w)
    else:

        if INITIAL:
            wn = tf.get_variable('Wn', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'scale_fc'], initializer=value[name])
        else:
            wn = tf.get_variable('Wn', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'scale_fc'], initializer=0.1)

        #tf.scalar_summary(wp.name, wp)
        tf.summary.scalar(wn.name, wn)

        mask = tf.ones(shape)
        mask_p = tf.where(x > eta_x, tf.ones(shape) * wn, mask)
        mask_np = tf.where(x < -eta_x, tf.ones(shape) * wn, mask_p)
        
        if binary:
            mask_z = tf.ones(shape)
        else:
            mask_z = tf.where((x < eta_x) & (x > - eta_x), tf.zeros(shape), tf.ones(shape))

        with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
            w =  tf.sign(x) * tf.stop_gradient(mask_z)

        w = w * mask_np

        tf.summary.histogram(w.name, w)


    return w

def rows_quant(x, eta, name, INITIAL, value, binary=False):

    shape = x.get_shape()

    eta_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * eta)

    list_of_masks = []

    if 'conv' in name:
        if INITIAL:
            w_s = tf.get_variable('Ws', [shape[0].value,1], collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'SCALING'], initializer=tf.constant_initializer(value[name]))
        else:
            w_s = tf.get_variable('Ws', [shape[0].value,1], collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'SCALING'], initializer=tf.constant_initializer(1.0))

        # for i in range(0,(shape[0].value*shape[1].value)):
            # tf.summary.scalar(w_s.name + str(i) +str(0), w_s[i,0])
            # tf.summary.scalar(w_s.name + str(i) + str(1), w_s[i, 1])
        
        #each row
        for i in range(shape[0].value):
            ws = w_s[i , 0]
            mask = tf.ones(shape)
            mask_p = tf.where(x[i,:,:,:] > eta_x, mask[i,:,:,:] * ws, mask[i,:,:,:])
            mask_np = tf.where(x[i,:,:,:] < -eta_x, mask[i,:,:,:] * ws, mask_p)
            list_of_masks.append(mask_np)

        masker = tf.stack(list_of_masks)
        masker = tf.reshape(masker, [i.value for i in shape])

        if binary:
            mask_z = tf.ones(shape)
        else:
            mask_z = tf.where((x < eta_x) & (x > - eta_x), tf.zeros(shape), tf.ones(shape))

        with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
            w =  tf.sign(x) * tf.stop_gradient(mask_z)

        w = w * masker

        tf.summary.histogram(w.name, w)

    else:
        if INITIAL:
            wn = tf.get_variable('Wn', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'scale_fc'], initializer=value[name])
        else:
            wn = tf.get_variable('Wn', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'scale_fc'], initializer=1.0)

        tf.summary.scalar(wn.name, wn)

        mask = tf.ones(shape)
        mask_p = tf.where(x > eta_x, tf.ones(shape) * wn, mask)
        mask_np = tf.where(x < -eta_x, tf.ones(shape) * wn, mask_p)
        
        if binary:
            mask_z = tf.ones(shape)
        else:
            mask_z = tf.where((x < eta_x) & (x > - eta_x), tf.zeros(shape), tf.ones(shape))

        with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
            w =  tf.sign(x) * tf.stop_gradient(mask_z)

        w = w * mask_np

        tf.summary.histogram(w.name, w)

    return w
