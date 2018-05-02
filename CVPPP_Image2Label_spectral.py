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
    # #FusionNet
    # @auto_reuse_variable_scope
    # def generator(self, img, last_dim=1, nl=tf.nn.tanh, nb_filters=32):
    #     assert img is not None
    #     return arch_generator(img, last_dim=last_dim, nl=nl, nb_filters=nb_filters)
    @staticmethod
    def build_res_block(x, name, chan, first=False):
        with tf.variable_scope(name):
            input = x
            return (LinearWrap(x)
                    .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
                    .Conv2D('conv0', chan, 3, padding='VALID')
                    .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
                    .Conv2D('conv1', chan, 3, padding='VALID', activation=tf.identity)
                    .InstanceNorm('inorm')()) + input

    @auto_reuse_variable_scope
    def generator(self, img):
        assert img is not None
        with argscope([Conv2D, Conv2DTranspose], activation=INReLU):
            l = (LinearWrap(img)
                 .tf.pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='SYMMETRIC')
                 .Conv2D('conv0', NF, 7, padding='VALID')
                 .Conv2D('conv1', NF * 2, 3, strides=2)
                 .Conv2D('conv2', NF * 4, 3, strides=2)())
            for k in range(9):
                l = Model.build_res_block(l, 'res{}'.format(k), NF * 4, first=(k == 0))
            l = (LinearWrap(l)
                 .Conv2DTranspose('deconv0', NF * 2, 3, strides=2)
                 .Conv2DTranspose('deconv1', NF * 1, 3, strides=2)
                 .tf.pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='SYMMETRIC')
                 .Conv2D('convlast', 16, 7, padding='VALID', activation=tf.tanh, use_bias=True)())
        return l
    @auto_reuse_variable_scope
    def discriminator(self, img):
        assert img is not None
        return arch_discriminator(img)


    def inputs(self):
        return [
            tf.placeholder(tf.float32, (DIMZ, DIMY, DIMX, 1), 'image'),
            tf.placeholder(tf.float32, (DIMZ, DIMY, DIMX, 1), 'membr'),
            tf.placeholder(tf.float32, (DIMZ, DIMY, DIMX, 1), 'label'),
            ]

    def build_graph(self, image, membr, label):
        G = tf.get_default_graph()
        with G.gradient_override_map({"Round": "Identity", "ArgMax": "Identity"}):
            pi, pa, pl = image, membr, label

            with tf.variable_scope('gen'):
                with tf.device('/device:GPU:0'):
                    with tf.variable_scope('image2label'):
                        pil = self.generator(tf_2tanh(pi), 
                                                 # last_dim=1, 
                                                 # nl=tf.nn.tanh, 
                                                 # nb_filters=32
                                                 )
                        # pil = tf_2imag(pil, maxVal=255.0)
                        # avg, var = tf.nn.moments(pil, axes=[1,2], keep_dims=True)
                        # pil -= avg
                        # pil /= (var+1e-6)
            losses = []         
            
            # with tf.name_scope('loss_mae'):
            #   mae_il = tf.reduce_mean(tf.abs(pl - tf_2imag(pil)), name='mae_il')
            #   losses.append(1e0*mae_il)
            #   add_moving_summary(mae_il)

            with tf.name_scope('loss_discrim'):
                delta_v     = 1.0 #0.5 #1.0 #args.dvar
                delta_d     = 3.0 #1.5 #3.0 #args.ddist
                param_var   = 1.0 #args.var
                param_dist  = 1.0 #args.dist
                param_reg   = 0.001 #args.reg
                #discrim_loss  =  ### Optimization operations
                # print pid
                # pid = tf.nn.softmax(pid)
                discrim_loss, _, _, _ = discriminative_loss_single(pil, 
                                                         pl, 
                                                         16,            # Feature dim
                                                         (DIMZ, DIMY, DIMX),    # Label shape
                                                         delta_v, 
                                                         delta_d, 
                                                         param_var, 
                                                         param_dist, 
                                                         param_reg)
                # cluster = L2Clustering()
                # discrim_loss = cluster.discriminative_loss(pl, pil)

                losses.append(1e-2*discrim_loss)
                add_moving_summary(discrim_loss)
            # Collect the result
            # pid = tf_2imag(pid)
            pil = tf.identity(pil, name='pil')
            self.cost = tf.reduce_sum(losses, name='self.cost')
            add_moving_summary(self.cost)
            # Visualization

            # Segmentation
            pz = tf.zeros_like(pi)
            viz = tf.concat([tf.concat([pi, 20*pl, 
                                            # 20*tf_2imag(pil), 
                                            128*pil[...,0:1], 128*pil[...,1:2], 128*pil[...,2:3]], axis=2),
                             # tf.concat([pz,   pz, pad[...,0:1], pad[...,1:2], pad[...,2:3]], axis=2),
                             # tf.concat([pz, 5*pal, 255*pala[...,0:1], 255*pala[...,1:2], 255*pala[...,2:3]], axis=2),
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
        from sklearn.cluster import MeanShift, estimate_bandwidth
        from sklearn.cluster import DBSCAN, SpectralClustering
        from sklearn import cluster

        self.dset = input 
        self._tower_name = tower_name
        self._device = device
        # self.algorithm = DBSCAN(eps=0.5)
        # self.algorithm = DBSCAN()
        # self.algorithm = cluster.AffinityPropagation(damping=.9, preference=-200)
        # bandwidth = cluster.estimate_bandwidth(X_flatten, quantile=0.2)
        # self.algorithm = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        # self.algorithm = cluster.SpectralClustering(
        #         n_clusters=n_clusters, eigen_solver='arpack',
        #         affinity="nearest_neighbors")
        
        # bandwidth = cluster.estimate_bandwidth(X_flatten, quantile=0.3)
        self.bandwidth=1.0
        # self.algorithm = cluster.MeanShift(bandwidth, n_jobs=n_jobs)
        self.algorithm = cluster.MeanShift(bandwidth= self.bandwidth, bin_seeding=True, n_jobs=4)
        # self.algorithm = SpectralClustering(n_clusters=n_clusters,
        #                         eigen_solver=None, random_state=None,
        #                         n_init=10, gamma=1.0, affinity='rbf',
        #                         n_neighbors=10, eigen_tol=0.0,
        #                         assign_labels='discretize', degree=3,
        #                         coef0=1,
        #                         kernel_params=None,
        #                         n_jobs=n_jobs)

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image', 'membr', 'label'], ['pil', 'viz'])

    def _before_train(self):
        pass

    def _trigger(self):
        for lst in self.dset.get_data():
            image, membr, label = lst
            print image.shape
            print membr.shape
            print label.shape
            pil, viz = self.pred(lst)
            pil = np.squeeze(np.array(pil))

            print pil.shape
          
            def np_func(X, algorithm, feature_dim=None, label_shape=None, n_clusters=12, n_jobs=4):
                # Perform clustering on high dimensional channel image
                feats_shape = X.shape
                if feature_dim==None:
                    feature_dim = feats_shape[-1]
                    print 'Feature_dim', feature_dim
                
                # Flatten and normalize X
                X_flatten = np.reshape(X, newshape=(-1, feature_dim))
                avg = X_flatten.mean()
                std = X_flatten.std()
                # X_flatten -= avg
                # X_flatten /= std
                print(X.shape)
                print(X_flatten.shape)
               

                print ('Perform clustering, might take some time ...')
                tic = time.time()
                algorithm.fit(X_flatten)
                # y_pred_flatten = algorithm.fit_predict(X_flatten)
                print ('Time for clustering', time.time() - tic)


                # Get the result in float32
                y_pred_flatten = algorithm.labels_.astype(np.float32)
                y_pred = np.reshape(y_pred_flatten, label_shape)

                print(label_shape)
                print(y_pred_flatten.shape)
                print(X_flatten.max())
                print(X_flatten.min())
                print(X_flatten.mean())
                print(X_flatten.std())
                print(y_pred_flatten.max())
                print(y_pred_flatten.min())


                return y_pred

            n_clusters = len(np.unique(label))
            preds = np_func(pil, self.algorithm, feature_dim=None, label_shape=[DIMZ, DIMY, DIMX, 1], n_clusters=n_clusters)    
          
            self.trainer.monitors.put_image('preds', 20*preds.astype(np.uint8))
            self.trainer.monitors.put_image('viz', viz)
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


    # train_ds  = PrefetchDataZMQ(train_ds, 4)
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
