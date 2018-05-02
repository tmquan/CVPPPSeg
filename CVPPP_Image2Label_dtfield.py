from Utility import * 

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

# Tensorlayer
from tensorlayer.cost import binary_cross_entropy, absolute_difference_error, dice_coe, cross_entropy

# Sklearn
from sklearn.metrics.cluster import adjusted_rand_score
###############################################################################
MAX_LABEL=320
DIMZ = 1
DIMY = 512
DIMX = 512


class Model(ModelDesc):
    #FusionNet
    @auto_reuse_variable_scope
    def generator(self, img, last_dim=1, nl=tf.nn.tanh, nb_filters=32):
        assert img is not None
        return arch_generator(img, last_dim=last_dim, nl=nl, nb_filters=nb_filters)
        # from enet import *
        # num_initial_blocks = 1
        # skip_connections = False
        # stage_two_repeat = 2
        # slim = tf.contrib.slim

        # with slim.arg_scope(ENet_arg_scope()):
        #     sfm, last_prelu = ENet(img,
        #                  num_classes=last_dim,
        #                  batch_size=DIMZ,
        #                  is_training=True,
        #                  reuse=None,
        #                  num_initial_blocks=num_initial_blocks,
        #                  stage_two_repeat=stage_two_repeat,
        #                  skip_connections=skip_connections)
        # return sfm, last_prelu
        # return arch_fusionnet(img)

    @auto_reuse_variable_scope
    def discriminator(self, img):
        assert img is not None
        return arch_discriminator(img)


    def inputs(self):
        return [
            tf.placeholder(tf.float32, (DIMZ, DIMY, DIMX, 1), 'image'),
            tf.placeholder(tf.float32, (DIMZ, DIMY, DIMX, 1), 'level'),
            ]

    def build_graph(self, image, level):
        G = tf.get_default_graph()
        with G.gradient_override_map({"Round": "Identity", "ArgMax": "Identity"}):
            pi, pl = image, level



            with tf.variable_scope('gen'):
                with tf.device('/device:GPU:0'):
                    with tf.variable_scope('image2level'):
                        pil, _  = self.generator(tf_2tanh(pi), 
                                                 last_dim=1, 
                                                 nl=INLReLU, 
                                                 nb_filters=32)
                      
            losses = []         
            
            with tf.name_scope('loss_mae'):
                mae_il = tf.reduce_mean(tf.abs(pl - 
                                             #tf.cast(tf.cast(pil, tf.int32), tf.float32)), 
                                             pil),
                                        name='mae_il')
                losses.append(1e0*mae_il)
                add_moving_summary(mae_il)

         
            # Collect the result
            pil = tf.identity(pil, name='pil')
            self.cost = tf.reduce_sum(losses, name='self.cost')
            add_moving_summary(self.cost)
            # Visualization

            # Segmentation
            pz = tf.zeros_like(pi)
            viz = tf.concat([tf.concat([pi, 20*pl,  20*pil], axis=2),
                             ], axis=1)
            # viz = tf_2imag(viz)
            viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
            tf.summary.image('labelized', viz, max_outputs=50)

    def optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)

###############################################################################
class VisualizeRunner(Callback):
    def __init__(self, input, tower_name='InferenceTower', device=0):
        self.dset = input 
        self._tower_name = tower_name
        self._device = device
        
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image', 'level'], ['viz'])

    def _before_train(self):
        pass

    def _trigger(self):
        for lst in self.dset.get_data():
            image, level = lst
            viz = self.pred(lst)
            self.trainer.monitors.put_image('viz', viz)
###############################################################################
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
            
            field_p = self.label2field(label_p)

            # Plus the label
            level_p = field_p + label_p

            #Expand dim to make single channel
            image_p = np.expand_dims(image_p, axis=-1)
            # membr_p = np.expand_dims(membr_p, axis=-1)
            label_p = np.expand_dims(label_p, axis=-1)
            level_p = np.expand_dims(level_p, axis=-1)

            #Return the membrane
            # membr_p = np_seg_to_aff(label_p)
            # membr_p = image_p.copy()

            yield [image_p.astype(np.float32), 
                   level_p.astype(np.float32), 
                   ]

    def label2field(self, labels):
        fields = np.zeros_like(labels)
        lst_labels, cnt_labels = np.unique(labels, return_counts=True)
        for label in lst_labels:
            field = np.zeros_like(labels)
            
            field[labels==label]  = 1.0 # Mask the current label

            #Perform distance transform
            from scipy import ndimage
            field = ndimage.distance_transform_edt(field)

            # Normalize:
            field = field/field.max() / 2.0 # From 0-1 to 0-0.5

            # Append to the final result
            fields = fields + field

        return fields

###############################################################################
def get_data(dataDir, isTrain=False, isValid=False, isTest=False):
    # Process the directories 
    if isTrain:
        num=500
        names = ['trainA', 'trainB']
    if isValid:
        num=1
        names = ['trainA', 'trainB']
    if isTest:
        num=1
        names = ['validA', 'validB']

    
    dset  = CVPPPDataFlow(os.path.join(dataDir, names[0]),
                               os.path.join(dataDir, names[1]),
                               num, 
                               isTrain=isTrain, 
                               isValid=isValid, 
                               isTest =isTest, 
                               shape=[DIMZ, DIMY, DIMX], 
                               pruneLabel=1)
    dset.reset_state()
    return dset
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',        default='0', help='comma seperated list of GPU(s) to use.')
    parser.add_argument('--data',  default='data/Kasthuri15/3D/', required=True, 
                                    help='Data directory, contain trainA/trainB/validA/validB')
    parser.add_argument('--load',   help='Load the model path')
    parser.add_argument('--sample', help='Run the deployment on an instance',
                                    action='store_true')

    args = parser.parse_args()
    # python Exp_FusionNet2D_-VectorField.py --gpu='0' --data='arranged/'

    
    train_ds = get_data(args.data, isTrain=True, isValid=False, isTest=False)
    valid_ds = get_data(args.data, isTrain=False, isValid=True, isTest=False)
    # test_ds  = get_data(args.data, isTrain=False, isValid=False, isTest=True)


    train_ds  = PrefetchDataZMQ(train_ds, 4)
    train_ds  = PrintData(train_ds)
    # train_ds  = QueueInput(train_ds)
    model     = Model()

    os.environ['PYTHONWARNINGS'] = 'ignore'

    # Set the GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Running train or deploy
    if args.sample:
        # TODO
        # sample
        pass
    else:
        # Set up configuration
        # Set the logger directory
        logger.auto_set_dir()

        # Set up configuration
        config = TrainConfig(
            model           =   model, 
            dataflow        =   train_ds,
            callbacks       =   [
                PeriodicTrigger(ModelSaver(), every_k_epochs=50),
                PeriodicTrigger(VisualizeRunner(valid_ds), every_k_epochs=50),
                ScheduledHyperParamSetter('learning_rate', [(0, 2e-4), (100, 1e-4), (200, 1e-5), (300, 1e-6)], interp='linear')
                ],
            max_epoch       =   500, 
            session_init    =    SaverRestore(args.load) if args.load else None,
            )
    
        # Train the model
        launch_train_with_config(config, QueueInputTrainer())
