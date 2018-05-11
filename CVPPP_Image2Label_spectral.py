from Utility import *


###############################################################################
class EMDataFlow(ImageDataFlow):
    ###############################################################################
    def AugmentPair(self, src_image, src_label, pipeline, seed=None, verbose=False):
        np.random.seed(seed) if seed else np.random.seed(2015)
        # print(src_image.shape, src_label.shape, aug_image.shape, aug_label.shape) if verbose else ''
        if src_image.ndim==2:
            src_image = np.expand_dims(src_image, 0)
            src_label = np.expand_dims(src_label, 0)
        
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
            # Calculate membrane
            def membrane(label):
                membr = np.zeros_like(label)
                for z in range(membr.shape[0]):
                    membr[z,...] = 1-skimage.segmentation.find_boundaries(np.squeeze(label[z,...]), mode='thick') #, mode='inner'
                membr[label==0] = 0 
                return membr

            membr_p = membrane(label_p.copy())


            #Expand dim to make single channel
            image_p = np.expand_dims(image_p, axis=-1)
            label_p = np.expand_dims(label_p, axis=-1)
            membr_p = np.expand_dims(membr_p, axis=-1)

           
            yield [image_p.astype(np.float32), 
                   membr_p.astype(np.float32), 
                   label_p.astype(np.float32), 
                   ] 

###############################################################################
class Model(ModelDesc):
    @auto_reuse_variable_scope
    def generator(self, image, last_dim=1, nl=INLReLU, nb_filters=32):
        assert image is not None
        return arch_fusionnet_2d(image, last_dim=last_dim, nl=nl, nb_filters=nb_filters)

    def inputs(self):
        return [
            tf.placeholder(tf.float32, (args.DIMZ, args.DIMY, args.DIMX, 1), 'image'),
            tf.placeholder(tf.float32, (args.DIMZ, args.DIMY, args.DIMX, 1), 'membr'),
            tf.placeholder(tf.float32, (args.DIMZ, args.DIMY, args.DIMX, 1), 'label'),
            ]

    def build_graph(self, image, membr, label):
        G = tf.get_default_graph()
        pi, pm, pl = image, membr, label

        feature_dim=32
        # Construct the graph
        with G.gradient_override_map({"Round": "Identity", "ArgMax": "Identity"}):
            with tf.variable_scope('gen'):
                with tf.device('/device:GPU:0'):
                    # with tf.variable_scope('image2embeds'):
                    #     pid = self.generator(tf_2tanh(pi), last_dim=feature_dim+1, nl=tf.nn.tanh, nb_filters=32)
                    with tf.variable_scope('image2membrs'):
                        pim, pif = self.generator(tf_2tanh(pi), last_dim=1, nl=tf.nn.tanh, nb_filters=32)
                    # with tf.variable_scope('image2embeds'):
                        # pif = self.generator(tf.concat([pim, tf_2tanh(pi)], axis=-1), last_dim=feature_dim, nl=INLReLU, nb_filters=32)
                        # pif = self.generator(pim, last_dim=feature_dim, nl=INLReLU, nb_filters=32)
                        # pif = tf.nn.dropout(pif,     keep_prob=0.5)
                        # pif = spatial_dropout(pif, 0.5, None, 'drop')
                        # avg, var = tf.nn.moments(pif, axes=[0,1,2,3], keep_dims=True)
                        # pif -= avg
                        # pif /= (var+1e-6)
        # pid = tf_2imag(pid, maxVal=1.0)
        pif = tf.identity(pif, name='pif')
        pim = tf.identity(pim, name='pim')
        #                 pif, pim = self.generator(tf_2tanh(pi), last_dim=64, nl=tf.nn.tanh, nb_filters=32)

        pim = tf_2imag(pim, maxVal=1.0)
        pif = tf_2imag(pif, maxVal=1.0)
        # # pim = tf.identity(pid[...,0:1], name='pim')
        # # pif = tf.identity(pid[...,1::], name='pif')
        # Define loss hre
        losses = [] 

        with tf.name_scope('loss_aff'):
            aff_im = tf.identity(1.0 - dice_coe(pim, pm, axis=[0,1,2,3], loss_type='jaccard'), 
                                 name='aff_im')  
            losses.append(1e1*aff_im)
            add_moving_summary(aff_im)

        with tf.name_scope('loss_mae'):
            mae_im = tf.reduce_mean(tf.abs(pm - pim), name='mae_im')
            losses.append(1e0*mae_im)
            add_moving_summary(mae_im)

        with tf.name_scope('loss_discrim'):

            discrim_loss = supervised_clustering_loss(pif, 
                                                     pl, 
                                                     feature_dim,            # Feature dim
                                                     (args.DIMZ, args.DIMY, args.DIMX),    # Label shape
                                                        )

            losses.append(1e-1*discrim_loss)
            add_moving_summary(discrim_loss)
        # Aggregate final loss
        self.cost = tf.reduce_sum(losses, name='self.cost')
        add_moving_summary(self.cost)

        # Segmentation
        pz = tf.zeros_like(pi)
        viz = tf.concat([tf.concat([pi, 15*pl, 255*pm, 255*pim], axis=2),
                         tf.concat([255*pif[...,0:1], 255*pif[...,1:2], pif[...,2:3], pif[...,3:4]], axis=2),                         
                         ], axis=1)
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
            ['image', 'membr', 'label'], ['viz', 'pim', 'pif'])

    def _before_train(self):
        pass

    def _trigger(self):
        for lst in self.dset.get_data():
            viz, pim, pif = self.pred(lst)



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
                X_flatten -= avg
                X_flatten /= std
                print(X.shape)
                print(X_flatten.shape)
               

                print ('Perform clustering, might take some time ...')
                tic = time.time()
                algorithm.fit(X_flatten)
                # y_pred_flatten = algorithm.fit_predict(X_flatten)
                print ('Time for clustering', time.time() - tic)


                # Get the result in float32
                y_pred_flatten = algorithm.labels_.astype(np.float32)
                if label_shape:
                    y_pred = np.reshape(y_pred_flatten, label_shape)
                else:
                    y_pred = y_pred_flatten

                print(label_shape)
                print(y_pred_flatten.shape)
                print(X_flatten.max())
                print(X_flatten.min())
                print(X_flatten.mean())
                print(X_flatten.std())
                print(y_pred_flatten.max())
                print(y_pred_flatten.min())


                return y_pred

            # Having mask and high dimensional space, so what's next?
            feature_dim=16
            # First squeeze everything
            pim = np.squeeze(np.array(pim)) #2d image
            pif = np.squeeze(np.array(pif)) #2d image

            pim_flatten = np.reshape(pim, [-1])
            pif_flatten = np.reshape(pif, [-1, feature_dim])

            loc_1d = pim_flatten>0.5      # Find the location of semantic segmentation
            idx_1d = np.where(loc_1d)   # Find the indices of semantic segmentation

            #Filter the high dim space
            pif_masked_flatten = pif_flatten[loc_1d]

            # Cluster them
            from sklearn.cluster import MeanShift, estimate_bandwidth
            from sklearn.cluster import DBSCAN, SpectralClustering
            from sklearn import cluster

            bandwidth = 1.0 #
            # bandwidth = cluster.estimate_bandwidth(pif_masked_flatten, quantile=0.3)
            algorithm = cluster.MeanShift(bandwidth= bandwidth, bin_seeding=True, n_jobs=4)
            pil_masked_flatten = np_func(pif_masked_flatten, algorithm, 
                                         feature_dim=feature_dim)

            # Reshape the label
            pil_flatten = np.zeros_like(pim_flatten)
            pil_flatten[idx_1d] = pil_masked_flatten
            pil = np.reshape(pil_flatten, pim.shape)

            self.trainer.monitors.put_image('pim_test', pim)
            self.trainer.monitors.put_image('pil_test', np.expand_dims(get_colors(np.squeeze(pil), plt.cm.PiYG)), axis=0)
            viz = np.squeeze(np.array(viz))
            self.trainer.monitors.put_image('viz_test', viz)
###############################################################################
def get_data(dataDir, isTrain=False, isValid=False, isTest=False, shape=[16, 320, 320]):
    # Process the directories 
    if isTrain:
        num=500
        names = ['trainA', 'trainB']
    if isValid:
        num=1
        names = ['trainA', 'trainB']
    if isTest:
        num=10
        names = ['validA', 'validB']

    
    dset  = EMDataFlow(os.path.join(dataDir, names[0]),
                               os.path.join(dataDir, names[1]),
                               num, 
                               isTrain=isTrain, 
                               isValid=isValid, 
                               isTest =isTest, 
                               shape=shape, 
                               pruneLabel=True)
    dset.reset_state()
    return dset
###############################################################################
def sample(dataDir, model_path, prefix='.'):
    print("Starting...")
    # print(dataDir)
    imageFiles = glob.glob(os.path.join(dataDir, 'testA/*.png'))
    labelFiles = glob.glob(os.path.join(dataDir, 'testB/*.png'))
    
    imageFiles = natsorted(imageFiles)
    labelFiles = natsorted(labelFiles)
    def AugmentPair(src_image, src_label, pipeline, seed=None, verbose=False):
        np.random.seed(seed) if seed else np.random.seed(2015)
        # print(src_image.shape, src_label.shape, aug_image.shape, aug_label.shape) if verbose else ''
        if src_image.ndim==2:
            src_image = np.expand_dims(src_image, 0)
            src_label = np.expand_dims(src_label, 0)
        
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
            # Prune the label in here
            aug_label, _ = skimage.measure.label(aug_label.copy(), return_num=True)         

            aug_images.append(aug_image)
            aug_labels.append(aug_label)

       

        aug_images = np.array(aug_images).astype(np.float32)
        aug_labels = np.array(aug_labels).astype(np.float32)
        # print(aug_images.shape, aug_labels.shape)
        aug_images = np.expand_dims(aug_images, axis=-1)
        aug_labels = np.expand_dims(aug_labels, axis=-1)
        return aug_images, aug_labels



    
    p_total = Augmentor.Pipeline()
    p_total.resize(probability=1, width=args.DIMY, height=args.DIMX, resample_filter='NEAREST')



    predict_func = OfflinePredictor(PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_path),
        input_names=['image', 'label'],
        output_names=['pim', 'pif']))

    sbds = []
    for k in range(len(imageFiles)): #range(1): #(len(imageFiles)):
        image = skimage.io.imread(imageFiles[k])
        label = skimage.io.imread(labelFiles[k])
        image, label = AugmentPair(image.copy(), label.copy(), p_total, seed=None)
        print(image.shape)

        # image = np.expand_dims(image, axis=3)
       
        ### Start deployment
       

        pred_pim, pred_pif = predict_func(image, label)
        pred_pim = np.array(pred_pim)
        pred_pif = np.array(pred_pif)

        print pred_pim.shape
        print pred_pif.shape



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
            X_flatten -= avg
            X_flatten /= std
            print(X.shape)
            print(X_flatten.shape)
           

            print ('Perform clustering, might take some time ...')
            tic = time.time()
            algorithm.fit(X_flatten)
            # y_pred_flatten = algorithm.fit_predict(X_flatten)
            print ('Time for clustering', time.time() - tic)


            # Get the result in float32
            y_pred_flatten = algorithm.labels_.astype(np.float32)
            if label_shape:
                y_pred = np.reshape(y_pred_flatten, label_shape)
            else:
                y_pred = y_pred_flatten

            print(label_shape)
            print(y_pred_flatten.shape)
            print(X_flatten.max())
            print(X_flatten.min())
            print(X_flatten.mean())
            print(X_flatten.std())
            print(y_pred_flatten.max())
            print(y_pred_flatten.min())


            return y_pred

         # Having mask and high dimensional space, so what's next?
        
        feature_dim=16
        # First squeeze everything
        pim = np.squeeze(np.array(pred_pim)) #2d image
        pif = np.squeeze(np.array(pred_pif)) #2d image

        pim_flatten = np.reshape(pim, [-1])
        pif_flatten = np.reshape(pif, [-1, feature_dim])

        loc_1d = pim_flatten>0.5      # Find the location of semantic segmentation
        idx_1d = np.where(loc_1d)   # Find the indices of semantic segmentation

        #Filter the high dim space
        pif_masked_flatten = pif_flatten[loc_1d]

        # Cluster them
        from sklearn.cluster import MeanShift, estimate_bandwidth
        from sklearn.cluster import DBSCAN, SpectralClustering
        from sklearn import cluster

        bandwidth = 1.0 #
        # bandwidth = cluster.estimate_bandwidth(pif_masked_flatten, quantile=0.3)
        # algorithm = cluster.MeanShift(bandwidth= bandwidth, bin_seeding=True, n_jobs=4)
        algorithm = cluster.SpectralClustering(n_clusters=label.max(), eigen_solver='arpack', affinity="nearest_neighbors")

        pil_masked_flatten = np_func(pif_masked_flatten, algorithm, 
                                     feature_dim=feature_dim)

        # Reshape the label
        pil_flatten = np.zeros_like(pim_flatten)
        pil_flatten[idx_1d] = pil_masked_flatten
        pil = np.reshape(pil_flatten, pim.shape)

        

        sbd = calc_sbd(label, pil)
        print 'Sbd ', sbd
        sbds.append(sbd)

        label = np.squeeze(label)
        pil = np.squeeze(pil)
        skimage.io.imsave('result_spectral/groundtruth/{}.png'.format(k+1), get_colors(label, plt.cm.PiYG)) #plt.cm.PiYG))
        skimage.io.imsave('result_spectral/predict/{}.png'.format(k+1), get_colors(pil, plt.cm.PiYG)) #plt.cm.PiYG))
    


    mean_sbd = np.mean(sbds)
    print 'Mean sbds ', mean_sbd
    print("Ending...")
    return None
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',        default='0', help='comma seperated list of GPU(s) to use.')
    parser.add_argument('--data',  default='data/Kasthuri15/3D/', required=True, 
                                    help='Data directory, contain trainA/trainB/validA/validB')
    parser.add_argument('--load',   help='Load the model path')
    parser.add_argument('--DIMX',  type=int, default=320)
    parser.add_argument('--DIMY',  type=int, default=320)
    parser.add_argument('--DIMZ',  type=int, default=16)
    parser.add_argument('--sample', help='Run the deployment on an instance',
                                    action='store_true')
    global args
    args = parser.parse_args()
    
    # python Exp_FusionNet2D_-VectorField.py --gpu='0' --data='arranged/'

    
    train_ds = get_data(args.data, isTrain=True, isValid=False, isTest=False, shape=[args.DIMZ, args.DIMY, args.DIMX])
    valid_ds = get_data(args.data, isTrain=False, isValid=True, isTest=False, shape=[args.DIMZ, args.DIMY, args.DIMX])
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
        print("Deploy the data")
        sample(args.data, args.load, prefix='deploy_')
        # pass
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
                PeriodicTrigger(VisualizeRunner(valid_ds), every_k_epochs=5),
                ScheduledHyperParamSetter('learning_rate', [(0, 2e-4), (100, 1e-4), (200, 1e-5), (300, 1e-6)], interp='linear')
                ],
            max_epoch       =   2000, 
            session_init    =   SaverRestore(args.load) if args.load else None,
            )
    
        # Train the model
        launch_train_with_config(config, QueueInputTrainer())






