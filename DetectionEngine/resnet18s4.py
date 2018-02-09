from keras.models import Model
from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, TimeDistributed
from keras.optimizers import Adam, SGD, RMSprop
import json
import traceback

import base
import losses.frcnn
from layers.FixedBatchNormalization import FixedBatchNormalization
from layers.RoiPoolingConv import RoiPoolingConv

class ResNet18s4Network(base.NeuralNetwork):
    """ Implementation of ResNet 18 neural network
    """

    #region Constructors

    def __init__(self, context, loss, isTraining):
        # Inputs
        self.imageInput, self.roiInput, self.featureMapInput = None, None, None
        # Layers
        self.baseLayers, self.regionProposalLayers, self.classifierLayers = None, None, None
        # Models
        self.regionProposalModel, self.classifierModel, self.classifierModelOnly, self.completeModel = None, None, None, None
        
        self.loss = loss
        # Base init
        super(ResNet18s4Network, self).__init__(context, isTraining)

     #endregion

    #region Methods

    def initializeInputs(self, context, isTraining):
        # Image input shape (any width, any height, 3 channels (RGB))
        self.imageInput = Input(shape=(None, None, 3))

        # ROI Input shape
        numROIs = int(context.getConfig('NeuralNetwork', 'num_rois'))
        self.roiInput = Input(shape=(numROIs, 4))

        # Feature map input shape
        if isTraining:
            self.featureMapInput = None
        else:
            self.featureMapInput = Input(shape=(None, None, 64))


    def initializeLayers(self, context, isTraining):

        # Create base, region and classfier layers
        self.baseLayers = self.createBaseLayers(self.imageInput, isTraining)
        self.regionProposalLayers = self.createRPNLayers(self.baseLayers, context)

        # If training, clasifier uses base layers, else, use feature maps
        if isTraining:
            classifierInput = self.baseLayers
        else:
            classifierInput = self.featureMapInput

        self.classifierLayers = self.createClassifierLayers(classifierInput, context)


    def initializeModels(self, context, isTraining):
        
        # When training, feature maps (base layer output) is not used
        if isTraining:
            self.regionProposalModel = Model(self.imageInput, self.regionProposalLayers[:2])
            self.classifierModel = Model([self.imageInput, self.roiInput], self.classifierLayers)
            self.completeModel = Model([self.imageInput, self.roiInput], self.regionProposalLayers[:2] + self.classifierLayers)
        else:
            self.regionProposalModel = Model(self.imageInput, self.regionProposalLayers)
            self.classifierModel = Model([self.featureMapInput, self.roiInput], self.classifierLayers)
            self.classifierModelOnly = Model([self.featureMapInput, self.roiInput], self.classifierLayers)
        
        # Load weights path
        weights_path = context.getConfig('NeuralNetwork', 'weights_path')
        if not weights_path:
            weights_path = context.getConfig('NeuralNetwork', 'resnet_18_weights_path')

        self.loadWeigths(weights_path)

        # Compile model with desired optimizer and loss functions
        frcnnLoss = self.loss

        optimizerRPN = Adam(lr=1e-5)
        optimizerClassifier = Adam(lr=1e-5)

        
        if isTraining:
            self.regionProposalModel.compile(optimizer=optimizerRPN, loss=[frcnnLoss.rpnClassLoss(context), frcnnLoss.rpnRegressionLoss(context)])
            self.classifierModel.compile(optimizer=optimizerClassifier, loss=[frcnnLoss.classifierClassLoss(context), frcnnLoss.classifierRegressionLoss(context)], metrics={'dense_class_{}'.format(len(context.classCount)): 'accuracy'})
            self.completeModel.compile(optimizer='sgd', loss='mae')
        else:
            self.regionProposalModel.compile(optimizer='sgd', loss='mse')
            self.classifierModel.compile(optimizer='sgd', loss='mse')

    
    def createBaseLayers(self, imageInput, trainable):
        bn_axis = 3
        x = ZeroPadding2D((3, 3))(imageInput)

        x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable = trainable)(x)
        x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, [64, 64], stage=2, block='a', strides=(1, 1), trainable = trainable)
        x = self.identity_block(x, 3, [64, 64], stage=2, block='b', trainable = trainable)
        #x = self.identity_block(x, 3, [64, 64], stage=2, block='c', trainable = trainable)
        #x = self.identity_block(x, 3, [64, 64], stage=2, block='d', trainable = trainable)
        
        return x


    def createRPNLayers(self, baseLayers, context):
        # Get the total number of anchors
        numAnchors = len(json.loads(context.getConfig('NeuralNetwork', 'anchor_scales'))) * len(json.loads(context.getConfig('NeuralNetwork', 'anchor_ratios')))

        x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(baseLayers)
        
        # x_class = prediction on objectness (for each anchor)
        x_class = Convolution2D(numAnchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
        # x_regr = prediction on bounding box coords (for each anchor)
        x_regr = Convolution2D(numAnchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

        return [x_class, x_regr, baseLayers]


    def createClassifierLayers(self, baseLayers, context):
        # Number of regions of interes to process simoultaneously
        numROIs = int(context.getConfig('NeuralNetwork', 'num_rois'))

        # Exact number of classes needed for 1-hot enconding
        numClasses = len(context.classCount)
      
        pooling_regions = 56
        input_shape = (numROIs, 56, 56, 64)

        out_roi_pool = RoiPoolingConv(pooling_regions, numROIs)([baseLayers, self.roiInput])
        out = self.classifier_layers(out_roi_pool, input_shape, trainable=True)

        out = TimeDistributed(Flatten())(out)

        out_class = TimeDistributed(Dense(numClasses, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(numClasses))(out)
        # note: no regression target for bg class
        out_regr = TimeDistributed(Dense(4 * (numClasses-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(numClasses))(out)

        return [out_class, out_regr]


    def trainRPN(self, image):
        rpnLoss = None
        rpnLoss = self.regionProposalModel.train_on_batch(image.resizedImageData, image.trainingRegions)
        return rpnLoss


    def trainClassifier(self, image, ROIs, ROIclassification, ROIregression, selectedSamples):
        X = image.resizedImageData
        X2 = ROIs
        Y1 = ROIclassification
        Y2 = ROIregression

        classifierLoss = self.classifierModel.train_on_batch([X, X2[:, selectedSamples, :]], [Y1[:, selectedSamples, :], Y2[:, selectedSamples, :]])

        return classifierLoss


    def predictRPN(self, image, isTraining):
        if isTraining:
            return self.regionProposalModel.predict_on_batch(image.resizedImageData)
        else:
            return self.regionProposalModel.predict(image.resizedImageData)


    def predictClassifier(self, input, isTraining):
        if isTraining:
            raise NotImplementedError
        else:
            return self.classifierModel.predict(input)


    def loadWeigths(self, filepath):
        # Load weights for both RPN and classifier
        try:
            self.regionProposalModel.load_weights(filepath, by_name=True)
            self.classifierModel.load_weights(filepath, by_name=True)
        except Exception as e:
            print('Error when loading weights: {} \n{}'.format(e, traceback.format_exc()))

    def saveWeights(self, filepath):
        self.completeModel.save_weights(filepath)

    def getFeatureMapSize(self, inputWidth, inputHeight):
        zeroPadding = 6
        stride = 2
        filters = [7, 3]
        featureMapWidth = inputWidth + zeroPadding
        featureMapHeight = inputHeight + zeroPadding

        for filter in filters:
            featureMapWidth = (featureMapWidth - filter + stride) // stride
            featureMapHeight = (featureMapHeight - filter + stride) // stride

        return featureMapWidth, featureMapHeight

    def identity_block(self, input_tensor, kernel_size, filters, stage, block, trainable=True):

        nb_filter1, nb_filter2 = filters
    
        bn_axis = 3

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Convolution2D(nb_filter1, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2a', trainable=trainable)(input_tensor)
        x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
        x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)

        return x

    def identity_block_td(self, input_tensor, kernel_size, filters, stage, block, trainable=True):

        # identity block time distributed

        nb_filter1, nb_filter2 = filters
        bn_axis = 3

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = TimeDistributed(Convolution2D(nb_filter1, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
        x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
        x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)

        return x

    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):

        nb_filter1, nb_filter2 = filters
        bn_axis = 3

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Convolution2D(nb_filter1, (kernel_size, kernel_size), padding='same', strides=strides, name=conv_name_base + '2a', trainable=trainable)(input_tensor)
        x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
        x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

        shortcut = Convolution2D(nb_filter2, (1, 1), padding='same', strides=strides, name=conv_name_base + '1', trainable=trainable)(input_tensor)
        shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    def conv_block_td(self, input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):

        # conv block time distributed

        nb_filter1, nb_filter2 = filters
        bn_axis = 3

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = TimeDistributed(Convolution2D(nb_filter1, (kernel_size, kernel_size), padding='same', strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
        x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
        x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
  
        shortcut = TimeDistributed(Convolution2D(nb_filter2, (1, 1), padding='same', strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
        shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    def classifier_layers(self, x, input_shape, trainable=False):

        x = self.conv_block_td(x, 3, [128, 128], stage=3, block='a',strides=(2,2), input_shape=input_shape, trainable = trainable)
        x = self.identity_block_td(x, 3, [128, 128], stage=3, block='b', trainable = trainable)

        x = self.conv_block_td(x, 3, [256, 256], stage=4, block='a',strides=(2,2), input_shape=x.shape, trainable = trainable)
        x = self.identity_block_td(x, 3, [256, 256], stage=4, block='b', trainable = trainable)

        x = self.conv_block_td(x, 3, [512, 512], stage=5, block='a', strides=(2, 2), input_shape=x.shape, trainable=trainable)
        x = self.identity_block_td(x, 3, [512, 512], stage=5, block='b', trainable=trainable)

        x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

        return x

    #endregion