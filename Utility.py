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

import PIL
from PIL import Image
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

# Tensorflow 
import tensorflow as tf
from tensorpack.models.common import layer_register, VariableHolder
from tensorpack.tfutils.common import get_tf_version_number
from tensorpack.utils.argtools import shape2d, shape4d, get_data_format
from tensorpack.models.tflayer import rename_get_variable, convert_to_tflayer_args

# Tensorlayer
from tensorlayer.cost import binary_cross_entropy, absolute_difference_error, dice_coe

# Sklearn
from sklearn.metrics.cluster import adjusted_rand_score

# Augmentor
import Augmentor
###############################################################################
EPOCH_SIZE = 100
NB_FILTERS = 32   # channel size
NF = 32   # channel size

DIMX  = 320
DIMY  = 320
DIMZ  = 3
DIMC  = 1

MAX_LABEL = 100
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
def tf_rand_score (x1, x2):
    def np_func (x1, x2):
        ret = np.mean(1.0 - adjusted_rand_score (x1.flatten (), x2.flatten ()))
        return ret
    tf_func = tf.py_func(np_func, [x1,  x2], [tf.float64])
    ret = tf_func[0]
    ret = tf.cast(ret, tf.float32)
    return ret

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
# This function read the bin from binary and prune to label
def tf_bin2dec(x, threshold=0.5, name='bin2dec'):
    # bin_num  = tf.to_int32(x>threshold) # Note that gradients will not flow
    # cond = tf.less(x, threshold*tf.ones_like(x))
    # bin_num  = tf.where(cond, tf.zeros_like(x), tf.ones_like(x))
    bin_num  = tf.identity(x)
    num_bits = bin_num.shape[-1] # 9 
    dec_num  = tf.reduce_sum(tf.multiply(tf.pow(tf.constant([2.0]), 
                                                tf.cast(tf.range(num_bits), dtype=tf.float32)), 
                             bin_num), axis=-1, keepdims=True)
    return tf.cast(dec_num, tf.float32, name=name) # return label

###############################################################################
# FusionNet
@layer_register(log_shape=True)
def residual(x, chan, first=False):
    with argscope([Conv2D], nl=INLReLU, stride=1, kernel_shape=3):
        input = x
        return (LinearWrap(x)
                .Conv2D('conv1', chan, padding='SAME', dilation_rate=1)
                .Conv2D('conv2', chan, padding='SAME', dilation_rate=2)
                .Conv2D('conv4', chan, padding='SAME', dilation_rate=4)             
                .Conv2D('conv5', chan, padding='SAME', dilation_rate=8)
                .Conv2D('conv0', chan, padding='SAME', nl=tf.identity)
                .InstanceNorm('inorm')()) + input

###############################################################################
@layer_register(log_shape=True)
def Subpix2D(inputs, chan, scale=2, stride=1):
    with argscope([Conv2D], nl=INLReLU, stride=stride, kernel_shape=3):
        results = Conv2D('conv0', inputs, chan* scale**2, padding='SAME')
        old_shape = inputs.get_shape().as_list()
        # results = tf.reshape(results, [-1, chan, old_shape[2]*scale, old_shape[3]*scale])
        # results = tf.reshape(results, [-1, old_shape[1]*scale, old_shape[2]*scale, chan])
        if scale>1:
            results = tf.depth_to_space(results, scale, name='depth2space', data_format='NHWC')
        return results

###############################################################################
@layer_register(log_shape=True)
def residual_enc(x, chan, first=False):
    with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=3):
        x = (LinearWrap(x)
            # .Dropout('drop', 0.75)
            .Conv2D('conv_i', chan, stride=2) 
            .residual('res_', chan, first=True)
            .Conv2D('conv_o', chan, stride=1) 
            ())
        return x

###############################################################################
@layer_register(log_shape=True)
def residual_dec(x, chan, first=False):
    with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=3):
                
        x = (LinearWrap(x)
            .Subpix2D('deconv_i', chan, scale=1) 
            .residual('res2_', chan, first=True)
            .Subpix2D('deconv_o', chan, scale=2) 
            # .Dropout('drop', 0.75)
            ())
        return x

###############################################################################
@auto_reuse_variable_scope
def arch_generator(img, last_dim=1, nl=INLReLU, nb_filters=NB_FILTERS):
    assert img is not None
    with argscope([Conv2D, Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
        e0 = residual_enc('e0', img, nb_filters*1)
        e1 = residual_enc('e1',  e0, nb_filters*2)
        e2 = residual_enc('e2',  e1, nb_filters*4)

        e3 = residual_enc('e3',  e2, nb_filters*8)
        # e3 = Dropout('dr', e3, 0.5)

        d3 = residual_dec('d3',    e3, nb_filters*4)
        d2 = residual_dec('d2', d3+e2, nb_filters*2)
        d1 = residual_dec('d1', d2+e1, nb_filters*1)
        d0 = residual_dec('d0', d1+e0, nb_filters*1) 
        d0 = Dropout('dr', d0, 0.5)
        dd =  (LinearWrap(d0)
                .Conv2D('convlast', last_dim, kernel_shape=3, stride=1, padding='SAME', nl=nl, use_bias=True) ())
        return dd, d0

@auto_reuse_variable_scope
def arch_discriminator(img, nb_filters=NB_FILTERS):
    assert img is not None
    with argscope([Conv2D, Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
        img = Conv2D('conv0', img, nb_filters, nl=INLReLU)
        e0 = residual_enc('e0', img, nb_filters*1)
        # e0 = Dropout('dr', e0, 0.5)
        e1 = residual_enc('e1',  e0, nb_filters*2)
        e2 = residual_enc('e2',  e1, nb_filters*4)

        e3 = residual_enc('e3',  e2, nb_filters*8)

        ret = Conv2D('convlast', e3, 1, stride=1, padding='SAME', nl=tf.identity, use_bias=True)
        return ret


def time_seed ():
    seed = None
    while seed == None:
        cur_time = time.time ()
        seed = int ((cur_time - int (cur_time)) * 1000000)
    return seed

###############################################################################
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

        images = natsorted (glob.glob(self.imageDir + '/*.*'))
        labels = natsorted (glob.glob(self.labelDir + '/*.*'))
        print(images)
        print(labels)
        self.images = []
        self.labels = []
        self.data_seed = time_seed ()
        self.data_rand = np.random.RandomState(self.data_seed)
        self.rng = np.random.RandomState(999)
        for i in range (len (images)):
            image = images[i]
            self.images.append (skimage.io.imread (image))
        for i in range (len (labels)):
            label = labels[i]
            self.labels.append (skimage.io.imread (label))
            
        self.DIMZ = shape[0]
        self.DIMY = shape[1]
        self.DIMX = shape[2]
        self.pruneLabel = pruneLabel

    def size(self):
        return self._size

    ###############################################################################
    def random_reverse(self, image, seed=None):
        assert ((image.ndim == 2) | (image.ndim == 3))
        if seed:
            self.rng.seed(seed)
        random_reverse = self.rng.randint(1,3)
        if random_reverse==1:
            reverse = image[::1,...]
        elif random_reverse==2:
            reverse = image[::-1,...]
        image = reverse
        return image
    
    

class CVPPPDataFlow(ImageDataFlow):
    ###############################################################################
    def AugmentPair(self, src_image, src_label, pipeline, seed=None, verbose=False):
        np.random.seed(seed) if seed else np.random.seed(2015)
        
        # print(src_image.shape, src_label.shape, aug_image.shape, aug_label.shape) if verbose else ''
        if src_image.ndim==2:
            src_image = np.expand_dims(src_image, 0)
            src_label = np.expand_dims(src_label, 0)
            # src_image = np.expand_dims(src_image, -1)
            # src_label = np.expand_dims(src_label, -1)
        
        # Create the result
        aug_images = [] #np.zeros_like(src_image)
        aug_labels = [] #np.zeros_like(src_label)
        
        # print(src_image.shape, src_label.shape)
        for z in range(src_image.shape[0]):
            #Image and numpy has different matrix order
            pipeline.set_seed(seed)
            aug_image = pipeline._execute_with_array(src_image[z,...]) 
            pipeline.set_seed(seed)
            aug_label = pipeline._execute_with_array(src_label[z,...])        
            aug_images.append(aug_image)
            aug_labels.append(aug_label)
        aug_images = np.array(aug_images).astype(np.float32)
        aug_labels = np.array(aug_labels).astype(np.float32)
        # print(aug_images.shape, aug_labels.shape)
        return aug_images, aug_labels
    ###############################################################################
    def ShuffleIndices(self, src_colors, num_colors=18, seed=None, verbose=False):
        np.random.seed(seed) if seed else np.random.seed(2015)
        
        # Unique unsort version
        _, idx_colors = np.unique(src_colors, return_index=True)
        #display(idx_colors)
        lst_colors = src_colors.flatten()[np.sort(idx_colors)]# np.unique(colors)
        # print(lst_colors) if verbose else ''
        num_colors = len(lst_colors)
        # print(num_colors) if verbose else ''
        
        # Take the label
        labels, num_labels = skimage.measure.label(src_colors, return_num=True)
        lst_labels, cnt_labels = np.unique(labels, return_counts=True)
        # print(lst_labels, num_labels, cnt_labels)  if verbose else ''
           
        # Permutation here
        aug_colors = np.zeros_like(src_colors)
        indices = np.arange(1, num_colors)
        shuffle = np.random.permutation(indices)
        # print(indices)  if verbose else ''
        # print(shuffle)  if verbose else ''

        for i, s in zip(indices, shuffle):
            aug_colors[src_colors==i] = s
        return aug_colors

    ###############################################################################
    def get_data(self):
        for k in range(self._size):
            #
            # Pick randomly a tuple of training instance
            #
            rand_index = self.data_rand.randint(0, len(self.images))
            image_p = self.images[rand_index].copy ()
            label_p = self.labels[rand_index].copy ()

            seed = time_seed () #self.rng.randint(0, 20152015)
            
            # Cut 1 or 3 slices along z, by define DIMZ, the same for paired, randomly for unpaired


            # dimz, dimy, dimx = image_p.shape
            # # The same for pair
            # randz = self.data_rand.randint(0, dimz-self.DIMZ+1)
            # randy = self.data_rand.randint(0, dimy-self.DIMY+1)
            # randx = self.data_rand.randint(0, dimx-self.DIMX+1)

            # image_p = image_p[randz:randz+self.DIMZ,randy:randy+self.DIMY,randx:randx+self.DIMX]
            # label_p = label_p[randz:randz+self.DIMZ,randy:randy+self.DIMY,randx:randx+self.DIMX]
            p_total = Augmentor.Pipeline()
            p_total.resize(probability=1, width=self.DIMY, height=self.DIMX, resample_filter='NEAREST')
            image_p, label_p = self.AugmentPair(image_p.copy(), label_p.copy(), p_total, seed=seed)
            

            if self.isTrain:
                # Augment the pair image for same seed
                p_train = Augmentor.Pipeline()
                p_train.rotate_random_90(probability=0.75, resample_filter=Image.NEAREST)
                p_train.rotate(probability=1, max_left_rotation=10, max_right_rotation=10, resample_filter=Image.NEAREST)
                p_train.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=5)
                # p_train.zoom_random(probability=0.5, percentage_area=0.8)
                p_train.flip_random(probability=0.75)

                image_p, label_p = self.AugmentPair(image_p.copy(), label_p.copy(), p_train, seed=seed)
                
                image_p = self.random_reverse(image_p, seed=seed)
                label_p = self.random_reverse(label_p, seed=seed)
                


            # # Calculate linear label
            if self.pruneLabel:
                label_p, nb_labels_p = skimage.measure.label(label_p.copy(), return_num=True)
        

            #Expand dim to make single channel
            image_p = np.expand_dims(image_p, axis=-1)
            # membr_p = np.expand_dims(membr_p, axis=-1)
            label_p = np.expand_dims(label_p, axis=-1)

            #Return the membrane
            # membr_p = np_seg_to_aff(label_p)
            membr_p = image_p.copy()

            yield [image_p.astype(np.float32), 
                   membr_p.astype(np.float32), 
                   label_p.astype(np.float32), 
                   ] 

# class SNEMI2DDataFlow(ImageDataFlow):
#     ###############################################################################
#     def AugmentPair(self, src_image, src_label, pipeline, seed=None, verbose=False):
#         np.random.seed(seed) if seed else np.random.seed(2015)
        
#         # print(src_image.shape, src_label.shape, aug_image.shape, aug_label.shape) if verbose else ''
#         if src_image.ndim==2:
#             src_image = np.expand_dims(src_image, 0)
#             src_label = np.expand_dims(src_label, 0)
#             # src_image = np.expand_dims(src_image, -1)
#             # src_label = np.expand_dims(src_label, -1)
        
#         # Create the result
#         aug_images = [] #np.zeros_like(src_image)
#         aug_labels = [] #np.zeros_like(src_label)
        
#         # print(src_image.shape, src_label.shape)
#         for z in range(src_image.shape[0]):
#             #Image and numpy has different matrix order
#             pipeline.set_seed(seed)
#             aug_image = pipeline._execute_with_array(src_image[z,...]) 
#             pipeline.set_seed(seed)
#             aug_label = pipeline._execute_with_array(src_label[z,...])        
#             aug_images.append(aug_image)
#             aug_labels.append(aug_label)
#         aug_images = np.array(aug_images).astype(np.float32)
#         aug_labels = np.array(aug_labels).astype(np.float32)
#         # print(aug_images.shape, aug_labels.shape)
#         return aug_images, aug_labels
    
#     ###############################################################################
#     def get_data(self):
#         for k in range(self._size):
#             #
#             # Pick randomly a tuple of training instance
#             #
#             rand_index = self.data_rand.randint(0, len(self.images))
#             image_p = self.images[rand_index].copy ()
#             label_p = self.labels[rand_index].copy ()

#             seed = time_seed () #self.rng.randint(0, 20152015)
            
#             # Cut 1 or 3 slices along z, by define DIMZ, the same for paired, randomly for unpaired


#             dimz, dimy, dimx = image_p.shape
#             # The same for pair
#             randz = self.data_rand.randint(0, dimz-self.DIMZ+1)
#             randy = self.data_rand.randint(0, dimy-self.DIMY+1)
#             randx = self.data_rand.randint(0, dimx-self.DIMX+1)

#             image_p = image_p[randz:randz+self.DIMZ,randy:randy+self.DIMY,randx:randx+self.DIMX]
#             label_p = label_p[randz:randz+self.DIMZ,randy:randy+self.DIMY,randx:randx+self.DIMX]
#             # p_total = Augmentor.Pipeline()
#             # p_total.resize(probability=1, width=self.DIMY, height=self.DIMX, resample_filter='NEAREST')
#             # image_p, label_p = self.AugmentPair(image_p.copy(), label_p.copy(), p_total, seed=seed)
            

#             if self.isTrain:
#                 # Augment the pair image for same seed
#                 p_train = Augmentor.Pipeline()
#                 p_train.rotate_random_90(probability=0.75, resample_filter=Image.NEAREST)
#                 p_train.rotate(probability=1, max_left_rotation=10, max_right_rotation=10, resample_filter=Image.NEAREST)
#                 p_train.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=5)
#                 # p_train.zoom_random(probability=0.5, percentage_area=0.8)
#                 p_train.flip_random(probability=0.75)

#                 image_p, label_p = self.AugmentPair(image_p.copy(), label_p.copy(), p_train, seed=seed)
                
#                 image_p = self.random_reverse(image_p, seed=seed)
#                 label_p = self.random_reverse(label_p, seed=seed)
                


#             # # Calculate linear label
#             if self.pruneLabel:
#                 label_p, nb_labels_p = skimage.measure.label(label_p.copy(), return_num=True)
        

#             #Expand dim to make single channel
#             image_p = np.expand_dims(image_p, axis=-1)
#             # membr_p = np.expand_dims(membr_p, axis=-1)
#             label_p = np.expand_dims(label_p, axis=-1)

#             #Return the membrane
#             # membr_p = np_seg_to_aff(label_p)
#             membr_p = image_p.copy()

#             yield [image_p.astype(np.float32), 
#                    membr_p.astype(np.float32), 
#                    label_p.astype(np.float32), 
#                    ] 



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


###############################################################################
@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def Conv3D(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        split=1):
    """
    A wrapper around `tf.layers.Conv2D`.
    Some differences to maintain backward-compatibility:
    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'.
    3. Support 'split' argument to do group conv.
    Variable Names:
    * ``W``: weights
    * ``b``: bias
    """
    if split == 1:
        with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
            layer = tf.layers.Conv3D(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer)
            ret = layer.apply(inputs, scope=tf.get_variable_scope())
            ret = tf.identity(ret, name='output')

        ret.variables = VariableHolder(W=layer.kernel)
        if use_bias:
            ret.variables.b = layer.bias

    else:
        # group conv implementation
        data_format = get_data_format(data_format, tfmode=False)
        in_shape = inputs.get_shape().as_list()
        channel_axis = 3 if data_format == 'NHWC' else 1
        in_channel = in_shape[channel_axis]
        assert in_channel is not None, "[Conv3D] Input cannot have unknown channel!"
        assert in_channel % split == 0

        assert kernel_regularizer is None and bias_regularizer is None and activity_regularizer is None, \
            "Not supported by group conv now!"

        out_channel = filters
        assert out_channel % split == 0
        assert dilation_rate == (1, 1) or get_tf_version_number() >= 1.5, 'TF>=1.5 required for group dilated conv'

        kernel_shape = shape2d(kernel_size)
        filter_shape = kernel_shape + [in_channel / split, out_channel]
        stride = shape4d(strides, data_format=data_format)

        kwargs = dict(data_format=data_format)
        if get_tf_version_number() >= 1.5:
            kwargs['dilations'] = shape4d(dilation_rate, data_format=data_format)

        W = tf.get_variable(
            'W', filter_shape, initializer=kernel_initializer)

        if use_bias:
            b = tf.get_variable('b', [out_channel], initializer=bias_initializer)

        inputs = tf.split(inputs, split, channel_axis)
        kernels = tf.split(W, split, 3)
        outputs = [tf.nn.conv2d(i, k, stride, padding.upper(), **kwargs)
                   for i, k in zip(inputs, kernels)]
        conv = tf.concat(outputs, channel_axis)
        if activation is None:
            activation = tf.identity
        ret = activation(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')

        ret.variables = VariableHolder(W=W)
        if use_bias:
            ret.variables.b = b
    return ret
###############################################################################
@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size', 'strides'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def Conv3DTranspose(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1, 1),
        padding='same',
        data_format='channels_last',
        activation=None,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None):
    """
    A wrapper around `tf.layers.Conv3DTranspose`.
    Some differences to maintain backward-compatibility:
    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'
    Variable Names:
    * ``W``: weights
    * ``b``: bias
    """

    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = tf.layers.Conv3DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer)
        ret = layer.apply(inputs, scope=tf.get_variable_scope())

    ret.variables = VariableHolder(W=layer.kernel)
    if use_bias:
        ret.variables.b = layer.bias
    return tf.identity(ret, name='output')


Deconv3D = Conv3DTranspose

auto_reuse_variable_scope
def arch_generator_3d(x, last_dim=1, nl=tf.nn.tanh, nb_filters=10, name='Vol3D_Encoder'):
    with argscope([Conv3D], kernel_shape=3, padding='SAME', nl=tf.nn.leaky_relu):
        with argscope([Conv3DTranspose], kernel_shape=3, padding='SAME', nl=tf.nn.leaky_relu):
            x = tf.expand_dims(x, axis=0) # to 1 256 256 256 1
            
        

            x1= Conv3D('conv1a', x ,  nb_filters*1, strides = 2, padding='SAME') #
            x2= Conv3D('conv2a', x1,  nb_filters*2, strides = 2, padding='SAME') #
            x3= Conv3D('conv3a', x2,  nb_filters*4, strides = 2, padding='SAME') #
            x4= Conv3D('conv4a', x3,  nb_filters*8, strides = 2, padding='SAME') #
            #x5= Conv3D('conv5a', x4, nb_filters*16, strides = 2, padding='SAME') #
            #x6= Conv3D('conv6a', x5, 512, strides = 2, padding='SAME') 
            
            z5=x4   
            #z6=Conv3DTranspose('conv6b', x6, 256, strides = 2, padding='SAME') 
            #z6=z6+x5 if x5 is not None else z6 
            #z5=Conv3DTranspose('conv5b', z6,  nb_filters*8, strides = 2, padding='SAME') 
            #z5=z5+x4 if x4 is not None else z5 
            z4=Conv3DTranspose('conv4b', z5,  nb_filters*4, strides = 2, padding='SAME') 
            z4=z4+x3 if x3 is not None else z4 
            z3=Conv3DTranspose('conv3b', z4,  nb_filters*2, strides = 2, padding='SAME') 
            z3=z3+x2 if x2 is not None else z3 
            z2=Conv3DTranspose('conv2b', z3,  nb_filters*1, strides = 2, padding='SAME') 
            z2=z2+x1 if x1 is not None else z2 
            z1=Conv3DTranspose('conv1b', z2,  last_dim, strides = 2, padding='SAME', nl=nl, use_bias=True) 
            
            z = tf.squeeze(z1, axis=0)

            return z

###############################################################################
class ClipCallback(Callback):
    def _setup_graph(self):
        vars = tf.trainable_variables()
        ops = []
        for v in vars:
            n = v.op.name
            if not n.startswith('discrim/'):
                continue
            logger.info("Clip {}".format(n))
            ops.append(tf.assign(v, tf.clip_by_value(v, -0.01, 0.01)))
        self._op = tf.group(*ops, name='clip')

    def _trigger_step(self):
        self._op.run()

###############################################################################

def get_center_loss(features, labels, feature_dim):
    with tf.variable_scope('center', reuse=True):
        centers = tf.get_variable('centers')
    features = tf.reshape(features, [-1, feature_dim])
    # len_features = features.get_shape()[1]
    labels = tf.reshape(labels, [-1])

    centers_batch = tf.gather(centers, labels)
    # Return the center loss
    loss = tf.reduce_sum((features - centers_batch) ** 2, [1])

    return loss

def update_centers(features, labels, feature_dim, alpha):
    with tf.variable_scope('center', reuse=True):
        centers = tf.get_variable('centers')
    features = tf.reshape(features, [-1, feature_dim])
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)

    diff = centers_batch - features

    # 
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff
    # 
    centers = tf.scatter_sub(centers,labels, diff)

    return centers
