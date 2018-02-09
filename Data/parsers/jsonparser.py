import os
import json
import copy

import entities.image

class Parser(object):
    """Class that reads training annotation files"""

    def getTrainingData(self, filePath):
        """Returns:
        [0] imageList: A list of TrainImage objects loaded from the file
        [1] clasMapping: Class indexes of all objects found 
        [2] classCount: Quantity of bounding boxes per class
        """
        imagesList = []
        classMapping = {}
        classCount = {'bg': 0}

        with open(filePath) as file:
            trainData = file.read()
            trainData = json.loads(trainData)
            
        trainDataRoot = trainData['rootPath']

        for trainImage in trainData['images']:
            newTrainImage = entities.image.TrainImage(os.path.join(trainDataRoot, trainImage['name']))
            newTrainImage.id = trainImage['name'].split('.')[0]
            newTrainImage.ext = trainImage['name'].split('.')[1]

            for className, boundingBoxes in trainImage['objects'].items():

                # If the object is new, add it to the class mapping dict
                if className not in classMapping and className != 'bg':
                    classMapping[className] = len(classMapping)
                    classCount[className] = 0

                # Count objects
                classCount[className] += len(boundingBoxes)

                # Load bounding boxes
                for bbox in boundingBoxes:
                    boundingBox = {'class': className, 'x1': int(bbox['x1']), 'x2': int(bbox['x2']), 'y1': int(bbox['y1']), 'y2': int(bbox['y2'])}
                    
                    # Test that bounding boxes values are correct
                    assert(boundingBox['x1'] < boundingBox['x2'] and boundingBox['y1'] < boundingBox['y2'])

                    newTrainImage.boundingBoxes.append(copy.deepcopy(boundingBox))
                    newTrainImage.originalBoundingBoxes.append(copy.deepcopy(boundingBox))

            imagesList.append(newTrainImage)

        # Add bg as last class index
        classMapping['bg'] = len(classMapping)

        return imagesList, classMapping, classCount

    def getSingleImageData(self, imageId):
        raise NotImplementedError()