import copy

class Image(object):
    """Class to store data related to a processed image"""

    #region Constructors

    def __init__(self, filePath):
        self.id = None
        self.ext = None
        self.filePath = filePath
        self.originalImageData = None
        self.imageData = None
        self.width = None
        self.height = None
        self.originalBoundingBoxes = [] # Keep original bounding boxes on a separate array
        self.boundingBoxes = []

        self.horizontalFlip = False
        self.verticalFlip = False
        self.rotation = 0

        self.ROIs = None
        self.predictedRegions = None # Placeholder for ImageRegionList
        self.classifiedRegions = None # Placeholder for ImageRegionList

    def cleanUp(self):
        self.originalImageData = None
        self.imageData = None
        self.width = None
        self.height = None
        self.boundingBoxes = copy.deepcopy(self.originalBoundingBoxes) # Reset boxes

        self.horizontalFlip = False
        self.verticalFlip = False
        self.rotation = 0

        self.ROIs = None
        self.predictedRegions = None
        self.classifiedRegions = None


    #endregion

class TrainImage(Image):
    """Extension of Image class for training purposes"""
    #region Constructors

    def __init__(self, filePath):
        super(TrainImage, self).__init__(filePath)
        self.trainingRegions = None 
        self.trainingROIs = None # Placeholder for TrainImageROI

    #endregion

class ImageRegionList(object):
    #region Constructors

    def __init__(self, classification, regression):
        self.classification = classification
        self.regression = regression

    #endregion

class TrainROIs(object):
    #region Constructors

    def __init__(self):
        self.ROIs = None
        self.classArray = None
        self.classLabels = None
        self.classCoordinates = None

    #endregion