from __future__ import print_function
from __future__ import absolute_import    

class NeuralNetwork(object):
    """Base Neural Network class"""

    #region Constructors

    def __init__(self, context, isTraining):

        # Initialize Input tensors for layers
        self.initializeInputs(context, isTraining)
        
        # Initialize neural network layers
        self.initializeLayers(context, isTraining)

        # Wrap layers in a model and compile (needed for executing the neural network)
        self.initializeModels(context, isTraining)
        

    #endregion

    #region Methods

    def initializeInputs(self, context, isTraining):
        raise NotImplementedError()

    def initializeLayers(self, context, isTraining):
        raise NotImplementedError()


    def initializeModels(self, context, isTraining):
        raise NotImplementedError()

    
    def trainRPN(self, imageData):
        raise NotImplementedError()


    def trainClassifier(self, imageData):
        raise NotImplementedError()


    def predictRPN(self, imageData):
        raise NotImplementedError()


    def predictClassifier(self, imageData):
        raise NotImplementedError()


    def loadWeigths(self):
        raise NotImplementedError()


    def saveWeights(self):
        raise NotImplementedError()

    def getFeatureMapSize(self, inputWidth, inputHeight):
        raise NotImplementedError()

    #endregion