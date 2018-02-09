from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.optimizers import Adam, SGD, RMSprop

import json
import base
import losses.frcnn
from layers.RoiPoolingConv import RoiPoolingConv

class VGG16Network(base.NeuralNetwork):
    """ Implementation of VGG 16 network
        Reference: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
        Code based from: https://github.com/yhenon/keras-frcnn
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
        super(VGG16Network, self).__init__(context, isTraining)

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
            self.featureMapInput = Input(shape=(None, None, 512))


    def initializeLayers(self, context, isTraining):

        # Create base, region and classfier layers
        self.baseLayers = self.createBaseLayers(self.imageInput, isTraining)
        self.regionProposalLayers = self.createRPNLayers(self.baseLayers, context)

        # If training, clasifier uses base layers, else, use feature maps
        if isTraining:
            classifierInput = self.baseLayers
        else:
            self.featureMapInput = self.featureMapInput

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
            self.completeModel = Model([self.imageInput, self.roiInput], self.regionProposalLayers + self.classifierLayers)
        
        # TODO: Are weigths really needed before compile?
        self.loadWeigths(context.getConfig('NeuralNetwork', 'vgg_weights_path'))

        # Compile model with desired optimizer and loss functions
        frcnnLoss = self.loss

        optimizerRPN = Adam(lr=1e-5)
        optimizerClassifier = Adam(lr=1e-5)

        self.regionProposalModel.compile(optimizer=optimizerRPN, loss=[frcnnLoss.rpnClassLoss(context), frcnnLoss.rpnRegressionLoss(context)])
        self.classifierModel.compile(optimizer=optimizerClassifier, loss=[frcnnLoss.classifierClassLoss(context), frcnnLoss.classifierRegressionLoss(context)], metrics={'dense_class_{}'.format(len(context.classCount)): 'accuracy'})
        self.completeModel.compile(optimizer='sgd', loss='mae')

    
    def createBaseLayers(self, imageInput, trainable):
        bn_axis = 3

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(imageInput)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        return x


    def createRPNLayers(self, baseLayers, context):
        # Get the total number of anchors
        numAnchors = len(json.loads(context.getConfig('NeuralNetwork', 'anchor_scales'))) * len(json.loads(context.getConfig('NeuralNetwork', 'anchor_ratios')))

        x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(baseLayers)

        x_class = Conv2D(numAnchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
        x_regr = Conv2D(numAnchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

        return [x_class, x_regr, baseLayers]


    def createClassifierLayers(self, baseLayers, context):
        # Number of regions of interes to process simoultaneously
        numROIs = int(context.getConfig('NeuralNetwork', 'num_rois'))

        # Exact number of classes needed for 1-hot enconding
        numClasses = len(context.classCount)
      
        pooling_regions = 14
        input_shape = (numROIs, 14, 14, 512)

        out_roi_pool = RoiPoolingConv(pooling_regions, numROIs)([baseLayers, self.roiInput])

        out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)

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


    def predictClassifier(self, imageData):
        raise NotImplementedError()


    def loadWeigths(self, filepath):
        # Load weights for both RPN and classifier
        self.regionProposalModel.load_weights(filepath, by_name=True)
        self.classifierModel.load_weights(filepath, by_name=True)

    def saveWeights(self, filepath):
        self.completeModel.save_weights(filepath)

    def getFeatureMapSize(self, inputWidth, inputHeight):
        return int(inputWidth / 16), int(inputHeight / 16)


    #endregion





