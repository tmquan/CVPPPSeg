# Hidden 2 domains no constrained
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, argparse, glob, time

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# Misc. libraries
from six.moves import map, zip, range
from natsort import natsorted 

# Array and image processing toolboxes
import numpy as np 
import skimage
import skimage.io
import skimage.transform
import skimage.segmentation
import malis

# For Augmentor
import PIL
from PIL import Image

# Augmentor
import Augmentor

# Tensorflow 
import tensorflow as tf

# Tensorpack toolbox
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.utils import get_rng
from tensorpack.tfutils import optimizer, gradproc
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary, add_tensor_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils import logger
from tensorpack.models.common import layer_register, VariableHolder
from tensorpack.tfutils.common import get_tf_version_number
from tensorpack.utils.argtools import shape2d, shape4d, get_data_format
from tensorpack.models.tflayer import rename_get_variable, convert_to_tflayer_args


# Tensorlayer
from tensorlayer.cost import binary_cross_entropy, absolute_difference_error, dice_coe, cross_entropy



###############################################################################
def INReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.relu(x, name=name)


def INLReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.leaky_relu(x, name=name)
    
def BNLReLU(x, name=None):
    x = BatchNorm('bn', x)
    return tf.nn.leaky_relu(x, name=name)
###############################################################################
def np_seg_to_aff(seg, nhood=malis.mknhood3d(1)):
    # return lambda seg, nhood: malis.seg_to_affgraph (seg, nhood).astype(np.float32)
    seg = np.squeeze(seg)
    seg = seg.astype(np.int32)
    ret = malis.seg_to_affgraph(seg, nhood) # seg zyx
    ret = ret.astype(np.float32)
    ret = np.squeeze(ret) # ret 3zyx
    ret = np.transpose(ret, [1, 2, 3, 0])# ret zyx3
    return ret
def tf_seg_to_aff(seg, nhood=tf.constant(malis.mknhood3d(1)), name='SegToAff'):
    # Squeeze the segmentation to 3D
    seg = tf.cast(seg, tf.int32)
    # Define the numpy function to transform segmentation to affinity graph
    # np_func = lambda seg, nhood: malis.seg_to_affgraph (seg, nhood).astype(np.float32)
    # Convert the numpy function to tensorflow function
    tf_func = tf.py_func(np_seg_to_aff, [seg, nhood], [tf.float32], name=name)
    # Reshape the result, notice that layout format from malis is 3, dimx, dimy, dimx
    # ret = tf.reshape(tf_func[0], [3, seg.shape[0], seg.shape[1], seg.shape[2]])
    # Transpose the result so that the dimension 3 go to the last channel
    # ret = tf.transpose(ret, [1, 2, 3, 0])
    # print seg.get_shape().as_list()
    ret = tf.reshape(tf_func[0], [seg.shape[0], seg.shape[1], seg.shape[2], 3])
    # print ret.get_shape().as_list()
    return ret
###############################################################################
def np_aff_to_seg(aff, nhood=malis.mknhood3d(1), threshold=np.array([0.5]) ):
    aff = np.transpose(aff, [3, 0, 1, 2]) # zyx3 to 3zyx
    ret = malis.connected_components_affgraph((aff > threshold[0]).astype(np.int32), nhood)[0].astype(np.int32) 
    ret = skimage.measure.label(ret).astype(np.float32)
    return ret
def tf_aff_to_seg(aff, nhood=tf.constant(malis.mknhood3d(1)), threshold=tf.constant(np.array([0.5])), name='AffToSeg'):
    # Define the numpy function to transform affinity to segmentation
    # def np_func (aff, nhood, threshold):
    #   aff = np.transpose(aff, [3, 0, 1, 2]) # zyx3 to 3zyx
    #   ret = malis.connected_components_affgraph((aff > threshold[0]).astype(np.int32), nhood)[0].astype(np.int32) 
    #   ret = skimage.measure.label(ret).astype(np.int32)
    #   return ret
    # print aff.get_shape().as_list()
    # Convert numpy function to tensorflow function
    tf_func = tf.py_func(np_aff_to_seg, [aff, nhood, threshold], [tf.float32], name=name)
    ret = tf.reshape(tf_func[0], [aff.shape[0], aff.shape[1], aff.shape[2]])
    ret = tf.expand_dims(ret, axis=-1)
    # print ret.get_shape().as_list()
    return ret


###############################################################################
# Utility function for scaling 
def tf_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
    with tf.variable_scope(name):
        return (x / maxVal - 0.5) * 2.0
        # x = tf.divide(x, tf.convert_to_tensor(maxVal))
        # x = tf.subtract(x, tf.convert_to_tensor(0.5))
        # x = tf.multiply(x, tf.convert_to_tensor(2.0))
        # return x
###############################################################################
def tf_2imag(x, maxVal = 255.0, name='ToRangeImag'):
    with tf.variable_scope(name):

        return (x / 2.0 + 0.5) * maxVal
        # x = tf.divide(x, tf.convert_to_tensor(2.0))
        # x = tf.add(x, tf.convert_to_tensor(0.5))
        # x = tf.multiply(x, tf.convert_to_tensor(maxVal))
        # return x

# Utility function for scaling 
def np_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
    return (x / maxVal - 0.5) * 2.0
###############################################################################
def np_2imag(x, maxVal = 255.0, name='ToRangeImag'):
    return (x / 2.0 + 0.5) * maxVal



###############################################################################
# @layer_register(log_shape=True)
# def Subpix2D(inputs, chan, scale=2, stride=1, kernel_shape=3):
#     with argscope([Conv2D], nl=INLReLU, stride=stride, kernel_shape=kernel_shape):
#         padded = tf.pad(inputs, paddings=[[0,0], [kernel_shape//2,kernel_shape//2], [kernel_shape//2,kernel_shape//2], [0,0]], 
#             mode='REFLECT', name='padded')
#         result = Conv2D('conv0', inputs, chan* scale**2, padding='VALID')
#         old_shape = inputs.get_shape().as_list()
#         if scale>1:
#             result = tf.depth_to_space(result, scale, name='depth2space', data_format='NHWC')
#         return result

###############################################################################
# FusionNet
@layer_register(log_shape=True)
def residual(x, chan, first=False, kernel_shape=3):
    with argscope([Conv2D], nl=INLReLU, stride=1, kernel_shape=kernel_shape):
        inputs = x
        # x = tf.pad(x, name='pad1', mode='REFLECT', paddings=[[0,0], [1*(kernel_shape//2),1*(kernel_shape//2)], [1*(kernel_shape//2),1*(kernel_shape//2)], [0,0]])
        # x = Conv2D('conv1', x, chan, padding='VALID', dilation_rate=1)
        # x = tf.pad(x, name='pad2', mode='REFLECT', paddings=[[0,0], [2*(kernel_shape//2),2*(kernel_shape//2)], [2*(kernel_shape//2),2*(kernel_shape//2)], [0,0]])
        # x = Conv2D('conv2', x, chan, padding='VALID', dilation_rate=2)
        # x = tf.pad(x, name='pad3', mode='REFLECT', paddings=[[0,0], [4*(kernel_shape//2),4*(kernel_shape//2)], [4*(kernel_shape//2),4*(kernel_shape//2)], [0,0]])
        # x = Conv2D('conv3', x, chan, padding='VALID', dilation_rate=4)             
        # x = tf.pad(x, name='pad4', mode='REFLECT', paddings=[[0,0], [8*(kernel_shape//2),8*(kernel_shape//2)], [8*(kernel_shape//2),8*(kernel_shape//2)], [0,0]])
        # x = Conv2D('conv4', x, chan, padding='VALID', dilation_rate=8)
        # x = tf.pad(x, name='pad0', mode='REFLECT', paddings=[[0,0], [kernel_shape//2,kernel_shape//2], [kernel_shape//2,kernel_shape//2], [0,0]])
        # x = Conv2D('conv0', x, chan, padding='VALID', nl=tf.identity)
        x = tf.pad(x, name='pad1', mode='REFLECT', paddings=[[0,0], [1*(kernel_shape//2),1*(kernel_shape//2)], [1*(kernel_shape//2),1*(kernel_shape//2)], [0,0]])
        x = Conv2D('conv1', x, chan, padding='VALID', dilation_rate=1)
        x = tf.pad(x, name='pad2', mode='REFLECT', paddings=[[0,0], [2*(kernel_shape//2),2*(kernel_shape//2)], [2*(kernel_shape//2),2*(kernel_shape//2)], [0,0]])
        x = Conv2D('conv2', x, chan, padding='VALID', dilation_rate=2)
        x = tf.pad(x, name='pad3', mode='REFLECT', paddings=[[0,0], [4*(kernel_shape//2),4*(kernel_shape//2)], [4*(kernel_shape//2),4*(kernel_shape//2)], [0,0]])
        x = Conv2D('conv3', x, chan, padding='VALID', dilation_rate=4)             
        # x = tf.pad(x, name='pad4', mode='REFLECT', paddings=[[0,0], [8*(kernel_shape//2),8*(kernel_shape//2)], [8*(kernel_shape//2),8*(kernel_shape//2)], [0,0]])
        # x = Conv2D('conv4', x, chan, padding='VALID', dilation_rate=8) 
        x = InstanceNorm('inorm', x) + inputs
        return x

###############################################################################
@layer_register(log_shape=True)
def residual_enc(x, chan, first=False, kernel_shape=3):
    with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=kernel_shape):
        
        x = tf.pad(x, name='pad_i', mode='REFLECT', paddings=[[0,0], [kernel_shape//2,kernel_shape//2], [kernel_shape//2,kernel_shape//2], [0,0]])
        x = Conv2D('conv_i', x, chan, stride=2) 
        x = residual('res_', x, chan, first=True)
        x = tf.pad(x, name='pad_o', mode='REFLECT', paddings=[[0,0], [kernel_shape//2,kernel_shape//2], [kernel_shape//2,kernel_shape//2], [0,0]])
        x = Conv2D('conv_o', x, chan, stride=1) 

        return x

###############################################################################
@layer_register(log_shape=True)
def residual_dec(x, chan, first=False, kernel_shape=3):
    with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=kernel_shape):
        x = Deconv2D('deconv_i', x, chan, stride=1) 
        x = residual('res2_', x, chan, first=True)
        x = Deconv2D('deconv_o', x, chan, stride=2) 

        return x

###############################################################################
@auto_reuse_variable_scope
def arch_fusionnet_2d(img, last_dim=1, nl=INLReLU, nb_filters=32):
    assert img is not None
    with argscope([Conv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='VALID'):
        with argscope([Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
            e0 = residual_enc('e0', img, nb_filters*1)
            e0 = Dropout('drop0', e0, 0.5)
            e1 = residual_enc('e1',  e0, nb_filters*2)
            e2 = residual_enc('e2',  e1, nb_filters*4)

            e3 = residual_enc('e3',  e2, nb_filters*8)
            e3 = Dropout('dr', e3, 0.5)

            d3 = residual_dec('d3',    e3, nb_filters*4)
            d2 = residual_dec('d2', d3+e2, nb_filters*2)
            d1 = residual_dec('d1', d2+e1, nb_filters*1)
            d1 = Dropout('drop1', d1, 0.5)
            d0 = residual_dec('d0', d1+e0, nb_filters*1) 

            dp = tf.pad( d0, name='pad_o', mode='REFLECT', paddings=[[0,0], [3//2,3//2], [3//2,3//2], [0,0]])
            dd = Conv2D('convlast', dp, last_dim, kernel_shape=3, stride=1, padding='VALID', nl=nl, use_bias=True) 
            return dd



###############################################################################
# tflearn
import tflearn
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.conv import conv_3d, conv_3d_transpose, max_pool_3d
from tflearn.layers.core import dropout
from tflearn.layers.merge_ops import merge
from tflearn.activations import linear, sigmoid, tanh, elu
from tensorflow.python.framework import ops

def tf_bottleneck(inputs, nb_filter, name="bottleneck"):
    with tf.variable_scope(name):
            original  = tf.identity(inputs, name="identity")

            with tf.contrib.framework.arg_scope([conv_3d, conv_3d_transpose], strides=[1, 1, 1, 1, 1], activation='leaky_relu'):
                    shape = original.get_shape().as_list()
                    conv_4x4i = original #conv_3d(incoming=original,    name="conv_4x4i", filter_size=4, nb_filter=nb_filter) # From 256 to 64 in Residual pape, bias=Falser
                    # original  = tf.nn.dropout(original, keep_prob=0.5)
                    conv_4x4i = conv_3d(incoming=conv_4x4i, name="conv_4x4i", filter_size=4, nb_filter=nb_filter, bias=False) # From 256 to 64 in Residual paper
                    # conv_4x4i = tf.nn.dropout(conv_4x4i, keep_prob=0.5)
                    conv_4x4m = conv_3d(incoming=conv_4x4i, name="conv_4x4m", filter_size=4, nb_filter=nb_filter, bias=False)
                    # conv_4x4o = tf.nn.dropout(conv_4x4o, keep_prob=0.5)
                    conv_4x4o = conv_3d(incoming=conv_4x4m, name="conv_4x4o", filter_size=4, nb_filter=nb_filter, bias=False, activation=tf.identity,
                                                                              # output_shape=[shape[1], shape[2], shape[3]]
                                                                              )
            summation = tf.add(original, conv_4x4o, name="summation")
            # summation = elu(summation)
            # return batch_normalization(summation)
            ret = InstanceNorm('bn', tf.squeeze(summation, axis=0))
            ret = tf.expand_dims(ret, axis=0)
            return ret


# In[10]:


def arch_fusionnet_3d(img, last_dim=1, nl=INLReLU, nb_filters=32, name='fusion3d'):
    # Add decorator to tflearn source code
    # sudo nano /usr/local/lib/python2.7/dist-packages/tflearn/layers/conv.py
    # @tf.contrib.framework.add_arg_scope
    with tf.variable_scope(name):
        with tf.contrib.framework.arg_scope([conv_3d], filter_size=4, strides=[1, 2, 2, 2, 1], activation='leaky_relu'):
            with tf.contrib.framework.arg_scope([conv_3d_transpose], filter_size=4, strides=[1, 2, 2, 2, 1], activation='leaky_relu'):
                shape = img.get_shape().as_list()
                dimb, dimz, dimy, dimx, dimc = shape
                e1a  = conv_3d(incoming=img,           name="e1a", nb_filter=nb_filters*1, bias=False)
                r1a  = tf_bottleneck(e1a,              name="r1a", nb_filter=nb_filters*1)
                r1a  = tf.nn.dropout(r1a,     keep_prob=0.5)

                e2a  = conv_3d(incoming=r1a,           name="e2a", nb_filter=nb_filters*1, bias=False)
                r2a  = tf_bottleneck(e2a,              name="r2a", nb_filter=nb_filters*1)
                r2a  = tf.nn.dropout(r2a,     keep_prob=0.5)

                e3a  = conv_3d(incoming=r2a,           name="e3a", nb_filter=nb_filters*2, bias=False)
                r3a  = tf_bottleneck(e3a,              name="r3a", nb_filter=nb_filters*2)
                r3a  = tf.nn.dropout(r3a,     keep_prob=0.5)

                e4a  = conv_3d(incoming=r3a,           name="e4a", nb_filter=nb_filters*2, bias=False)
                r4a  = tf_bottleneck(e4a,              name="r4a", nb_filter=nb_filters*2)
                r4a  = tf.nn.dropout(r4a,     keep_prob=0.5)

                e5a  = conv_3d(incoming=r4a,           name="e5a", nb_filter=nb_filters*4, bias=False)
                r5a  = tf_bottleneck(e5a,              name="r5a", nb_filter=nb_filters*4)
                r5a  = tf.nn.dropout(r5a,     keep_prob=0.5)

                # e6a  = conv_3d(incoming=r5a,           name="e6a", nb_filter=nb_filters*4, bias=False)
                # r6a  = tf_bottleneck(e6a,              name="r6a", nb_filter=nb_filters*4)

                # e7a  = conv_3d(incoming=r6a,           name="e7a", nb_filter=nb_filters*8)           , bias=False 
                # r7a  = tf_bottleneck(e7a,              name="r7a", nb_filter=nb_filters*8)
                # r7a  = dropout(incoming=r7a, keep_prob=0.5)
                print "In1 :", img.get_shape().as_list()
                print "E1a :", e1a.get_shape().as_list()
                print "R1a :", r1a.get_shape().as_list()
                print "E2a :", e2a.get_shape().as_list()
                print "R2a :", r2a.get_shape().as_list()
                print "E3a :", e3a.get_shape().as_list()
                print "R3a :", r3a.get_shape().as_list()
                print "E4a :", e4a.get_shape().as_list()
                print "R4a :", r4a.get_shape().as_list()
                print "E5a :", e5a.get_shape().as_list()
                print "R5a :", r5a.get_shape().as_list()
                
                r5b  = tf_bottleneck(r5a,              name="r5b", nb_filter=nb_filters*4)
                d4b  = conv_3d_transpose(incoming=r5b, name="d4b", nb_filter=nb_filters*2, output_shape=[-(-dimz//(2**4)), -(-dimy//(2**4)), -(-dimx/(2**4))], bias=False)
                a4b  = tf.add(d4b, r4a,            name="a4b")

                r4b  = tf_bottleneck(a4b,              name="r4b", nb_filter=nb_filters*2)
                d3b  = conv_3d_transpose(incoming=r4b, name="d3b", nb_filter=nb_filters*2, output_shape=[-(-dimz//(2**3)), -(-dimy//(2**3)), -(-dimx/(2**3))], bias=False)
                a3b  = tf.add(d3b, r3a,            name="a3b")


                r3b  = tf_bottleneck(a3b,              name="r3b", nb_filter=nb_filters*2)
                d2b  = conv_3d_transpose(incoming=r3b, name="d2b", nb_filter=nb_filters*1, output_shape=[-(-dimz//(2**2)), -(-dimy//(2**2)), -(-dimx/(2**2))], bias=False)
                a2b  = tf.add(d2b, r2a,            name="a2b")

                r2b  = tf_bottleneck(a2b,              name="r2b", nb_filter=nb_filters*1)
                d1b  = conv_3d_transpose(incoming=r2b, name="d1b", nb_filter=nb_filters*1, output_shape=[-(-dimz//(2**1)), -(-dimy//(2**1)), -(-dimx/(2**1))], bias=False)
                a1b  = tf.add(d1b, r1a,            name="a1b")

                out  = conv_3d_transpose(incoming=a1b, name="out", nb_filter=last_dim,
                                                                activation='tanh',
                                                                output_shape=[-(-dimz//(2**0)), -(-dimy//(2**0)), -(-dimx/(2**0))])


                # print "R7b :", r7b.get_shape().as_list()
                # print "D6b :", d6b.get_shape().as_list()
                # print "A6b :", a6b.get_shape().as_list()

                # print "R6b :", r6b.get_shape().as_list()
                # print "D5b :", d5b.get_shape().as_list()
                # print "A5b :", a5b.get_shape().as_list()

                print "R5b :", r5b.get_shape().as_list()
                print "D4b :", d4b.get_shape().as_list()
                print "A4b :", a4b.get_shape().as_list()

                print "R4b :", r4b.get_shape().as_list()
                print "D3b :", d3b.get_shape().as_list()
                print "A3b :", a3b.get_shape().as_list()

                print "R3b :", r3b.get_shape().as_list()
                print "D2b :", d2b.get_shape().as_list()
                print "A2b :", a2b.get_shape().as_list()

                print "R2b :", r2b.get_shape().as_list()
                print "D1b :", d1b.get_shape().as_list()
                print "A1b :", a1b.get_shape().as_list()

                print "Out :", out.get_shape().as_list()

                return out


###############################################################################

def time_seed ():
    seed = None
    while seed == None:
        cur_time = time.time ()
        seed = int ((cur_time - int (cur_time)) * 1000000)
    return seed

class ImageDataFlow(RNGDataFlow):
    def __init__(self, 
        imageDir, 
        labelDir, 
        size, 
        dtype='float32', 
        isTrain=False, 
        isValid=False, 
        isTest=False, 
        pruneLabel=False, 
        shape=[3, 320, 320]):

        self.dtype      = dtype
        self.imageDir   = imageDir
        self.labelDir   = labelDir
        self._size      = size
        self.isTrain    = isTrain
        self.isValid    = isValid

        imageFiles = natsorted (glob.glob(self.imageDir + '/*.*'))
        labelFiles = natsorted (glob.glob(self.labelDir + '/*.*'))
        print(imageFiles)
        print(labelFiles)
        self.images = []
        self.labels = []
        self.data_seed = time_seed ()
        self.data_rand = np.random.RandomState(self.data_seed)
        self.rng = np.random.RandomState(999)
        for imageFile in imageFiles:
            image = skimage.io.imread (imageFile)
            self.images.append(image)
        for labelFile in labelFiles:
            label = skimage.io.imread (labelFile)
            self.labels.append(label)
            
        self.DIMZ = shape[0]
        self.DIMY = shape[1]
        self.DIMX = shape[2]
        self.pruneLabel = pruneLabel

    def size(self):
        return self._size

   



def tf_norm(inputs, axis=1, epsilon=1e-7,  name='safe_norm'):
    squared_norm    = tf.reduce_sum(tf.square(inputs), axis=axis, keepdims=True)
    safe_norm       = tf.sqrt(squared_norm+epsilon)
    return tf.identity(safe_norm, name=name)

def discriminative_loss_single(prediction, correct_label, feature_dim, label_shape, 
                            delta_v, delta_d, param_var, param_dist, param_reg):
    
    ''' Discriminative loss for a single prediction/label pair.
    :param prediction: inference of network
    :param correct_label: instance label
    :feature_dim: feature dimension of prediction
    :param label_shape: shape of label
    :param delta_v: cutoff variance distance
    :param delta_d: curoff cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    '''

    ### Reshape so pixels are aligned along a vector
    correct_label = tf.reshape(correct_label,   [label_shape[2]*label_shape[1]*label_shape[0]])
    reshaped_pred = tf.reshape(prediction,      [label_shape[2]*label_shape[1]*label_shape[0], feature_dim])

    ### Count instances
    unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)
    counts = tf.cast(counts, tf.float32)
    num_instances = tf.size(unique_labels)

    segmented_sum = tf.unsorted_segment_sum(reshaped_pred, unique_id, num_instances)

    mu = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))
    mu_expand = tf.gather(mu, unique_id)

    ### Calculate l_var
    distance = tf_norm(tf.subtract(mu_expand, reshaped_pred), axis=1)
    distance = tf.subtract(distance, delta_v)
    distance = tf.clip_by_value(distance, 0., distance)
    distance = tf.square(distance)

    l_var = tf.unsorted_segment_sum(distance, unique_id, num_instances)
    l_var = tf.div(l_var, counts)
    l_var = tf.reduce_sum(l_var)
    l_var = tf.divide(l_var, tf.cast(num_instances, tf.float32))
    
    ### Calculate l_dist
    
    # Get distance for each pair of clusters like this:
    #   mu_1 - mu_1
    #   mu_2 - mu_1
    #   mu_3 - mu_1
    #   mu_1 - mu_2
    #   mu_2 - mu_2
    #   mu_3 - mu_2
    #   mu_1 - mu_3
    #   mu_2 - mu_3
    #   mu_3 - mu_3

    mu_interleaved_rep = tf.tile(mu, [num_instances, 1])
    mu_band_rep = tf.tile(mu, [1, num_instances])
    mu_band_rep = tf.reshape(mu_band_rep, (num_instances*num_instances, feature_dim))

    mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)
    
    # Filter out zeros from same cluster subtraction
    intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff),axis=1)
    zero_vector = tf.zeros(1, dtype=tf.float32)
    bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
    mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)

    mu_norm = tf_norm(mu_diff_bool, axis=1)
    mu_norm = tf.subtract(2.*delta_d, mu_norm)
    mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)
    mu_norm = tf.square(mu_norm)

    l_dist = tf.reduce_mean(mu_norm)

    ### Calculate l_reg
    l_reg = tf.reduce_mean(tf_norm(mu, axis=1))

    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    loss = param_scale*(l_var + l_dist + l_reg)
    
    return loss, l_var, l_dist, l_reg


def discriminative_loss(prediction, correct_label, feature_dim, image_shape, 
                delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Iterate over a batch of prediction/label and cumulate loss
    :return: discriminative loss and its three components
    '''
    def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
        disc_loss, l_var, l_dist, l_reg = discriminative_loss_single(prediction[i], correct_label[i], feature_dim, image_shape, 
                        delta_v, delta_d, param_var, param_dist, param_reg)

        out_loss = out_loss.write(i, disc_loss)
        out_var = out_var.write(i, l_var)
        out_dist = out_dist.write(i, l_dist)
        out_reg = out_reg.write(i, l_reg)

        return label, batch, out_loss, out_var, out_dist, out_reg, i + 1

    # TensorArray is a data structure that support dynamic writing
    output_ta_loss = tf.TensorArray(dtype=tf.float32,
                   size=0,
                   dynamic_size=True)
    output_ta_var = tf.TensorArray(dtype=tf.float32,
                   size=0,
                   dynamic_size=True)
    output_ta_dist = tf.TensorArray(dtype=tf.float32,
                   size=0,
                   dynamic_size=True)
    output_ta_reg = tf.TensorArray(dtype=tf.float32,
                   size=0,
                   dynamic_size=True)

    _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _  = tf.while_loop(cond, body, [correct_label, 
                                                        prediction, 
                                                        output_ta_loss, 
                                                        output_ta_var, 
                                                        output_ta_dist, 
                                                        output_ta_reg, 
                                                        0])
    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()
    
    disc_loss = tf.reduce_mean(out_loss_op)
    l_var = tf.reduce_mean(out_var_op)
    l_dist = tf.reduce_mean(out_dist_op)
    l_reg = tf.reduce_mean(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg




def supervised_clustering_loss(prediction, correct_label, feature_dim, label_shape):
    Y = tf.reshape(correct_label, [label_shape[1]*label_shape[2]*label_shape[0], 1])
    F = tf.reshape(prediction, [label_shape[1]*label_shape[2]*label_shape[0], feature_dim])
    diagy = tf.reduce_sum(Y,0)
    onesy = tf.ones(diagy.get_shape())
    J = tf.matmul(Y,tf.diag(tf.rsqrt(tf.where(tf.greater_equal(diagy,onesy),diagy,onesy))))
    [S,U,V] = tf.svd(F)
    Slength = tf.cast(tf.reduce_max(S.get_shape()), tf.float32)
    maxS = tf.fill(tf.shape(S),tf.scalar_mul(tf.scalar_mul(1e-15,tf.reduce_max(S)),Slength))
    ST = tf.where(tf.greater_equal(S,maxS),tf.div(tf.ones(S.get_shape()),S),tf.zeros(S.get_shape()))
    pinvF = tf.transpose(tf.matmul(U,tf.matmul(tf.diag(ST),V,False,True)))
    FJ = tf.matmul(pinvF,J)
    G = tf.matmul(tf.subtract(tf.matmul(F,FJ),J),FJ,False,True)
    loss = tf.reduce_sum(tf.multiply(tf.stop_gradient(G),F))
    return loss

