import numpy as np
import time
import traceback
import os
import importlib
import traceback
import inspect
import json
import pickle
import ast

import entities
import parsers.jsonparser
import processors.optimizedPreprocessor
import losses.frcnn
import base # Neural network base
import utils.drawing
import cv2

class Orchestrator(object):
    """Class that handles high level flow for Faster RCNN algorithm"""
    def __init__(self, logger):
        self.logger = logger
    #region Methods

    def train(self, context):
        # TODO: Make logging static
        self.logger.info('Starting Training')
                
        # Parse training annotations file
        self.logger.info('Training file: ' + context.dataPath)
        jsonParser = parsers.jsonparser.Parser()
        imagesList, classMapping, classCount = jsonParser.getTrainingData(context.dataPath)

        self.logger.info('Class mapping: ' + str(classMapping))
        self.logger.info('Class count: ' + str(classCount))

        # Save class map & count
        context.classCount = classCount
        context.setConfig('Training', 'class_mapping', str(classMapping))
        context.saveConfig()

        # An epoch is a full pass of all the dataset (Length = Number of images)
        epochs = int(context.getConfig('Training', 'num_epochs'))
        epochLength = len(imagesList)

        #Initialize loss class for calculations
        frcnnLoss = losses.frcnn.FasterRCNNLoss(epochLength)

        # Create Neural network
        # This initializes layers, compiles the model and loads weigths
        neuralNetwork = self.importNetwork(context, frcnnLoss, True)

        # Get data iterator used to cycle through training data samples
        preprocessor = processors.optimizedPreprocessor.OptimizedImagePreprocessor(self.logger, neuralNetwork)
        imageIterator = preprocessor.getImageDataIterator(imagesList, context)

        # Control variables to monitor training process
        bestLoss = np.inf
        rpnOverlapsInEpoch = []
        epochIteration = 0
        startTime = time.time()

        # Training Loop
        for epoch in range(epochs):
            self.logger.info('Epoch {}/{}'.format(epoch, epochs))

            # Training keeps going even when something fails
            # TODO: Add progbar
            while True:
                try:
                    # Get the next image
                    trainingImage = next(imageIterator)

                    # -----------------------------------
                    # ------- RPN training --------------
                    # -----------------------------------
                    self.logger.info('Training RPN on image')
                    rpnLoss = neuralNetwork.trainRPN(trainingImage)

                    # Predict on the same image
                    trainingImage.predictedRegions = neuralNetwork.predictRPN(trainingImage, True)

                    # Transform RPN results to ROI (Regions of interest)
                    self.logger.info('Transform RPN result to ROIs')
                    regions, probs = preprocessor.rpnToROI(trainingImage, context)
                    ROIs, ROIclassification, ROIregression = preprocessor.nonMaxSuppressionWithIOU(trainingImage, regions, probs, context)
                    self.logger.info('IOU regions: {}'.format(ROIs.shape))

                    utils.drawing.ImageDrawer().drawRegionsOnImageWH(trainingImage, ROIs, context, ROIclassification, str(epoch))

                    # If no ROIs where calculated, continue
                    if ROIs is None:
                        self.logger.info('No overlapping ROIs found, please check configurations')
                        rpnOverlapsInEpoch.append(0)
                        continue

                    # ----------------------------------
                    # ------- Clasifier training -------
                    # ----------------------------------
                    # We use a mask to keep track of the remaining samples, the idea is to pass each of them once through the network.
                    numROIs = int(context.getConfig('NeuralNetwork', 'num_rois'))
                    remainingSamples = np.ones(ROIs.shape[1]).astype(np.bool)
                    
                    # Keep training classifier until there are no more positive samples to pass
                    while True:
                        # Divide positive and negative samples
                        negSamples = np.where(np.logical_and(ROIclassification[0, :, -1] == 1, remainingSamples == True))[0] # Negative ROIS = BG
                        posSamples = np.where(np.logical_and(ROIclassification[0, :, -1] == 0, remainingSamples == True))[0] # Positive, all others
                        self.logger.info('Positive samples: {}'.format(len(posSamples)))
                        self.logger.info('Negative samples: {}'.format(len(negSamples)))
                        if len(posSamples) == 0: break
                        positiveSamples, negativeSamples, remainingSamples = self.selectROISamples(posSamples, negSamples, numROIs, remainingSamples)
                        if len(positiveSamples) == 0: break
                        selectedSamples = positiveSamples + negativeSamples
                    
                        # Train classifier
                        self.logger.info('Training classifier ROIs on samples')
                        classifierLoss = neuralNetwork.trainClassifier(trainingImage, ROIs, ROIclassification, ROIregression, selectedSamples)

                    # Track iteration losses
                    frcnnLoss.loss_rpn_cls[epochIteration] = rpnLoss[1]
                    frcnnLoss.loss_rpn_regr[epochIteration] = rpnLoss[2]
                    frcnnLoss.loss_class_cls[epochIteration] = classifierLoss[1]
                    frcnnLoss.loss_class_regr[epochIteration] = classifierLoss[2]
                    frcnnLoss.class_accuracy[epochIteration] = classifierLoss[3]

                    epochIteration += 1

                    # If epoch completed
                    if epochIteration == epochLength:
                        meanClassAccuracy = np.mean(frcnnLoss.class_accuracy)
                        meanLossRpnC = np.mean(frcnnLoss.loss_rpn_cls)
                        meanLossRpnR = np.mean(frcnnLoss.loss_rpn_regr)
                        meanLossClassC = np.mean(frcnnLoss.loss_class_cls)
                        meanLossClassR = np.mean(frcnnLoss.loss_class_regr)

                        self.logger.info('Classifier accuracy: {}'.format(meanClassAccuracy))
                        self.logger.info('Loss RPN classifier: {}'.format(meanLossRpnC))
                        self.logger.info('Loss RPN regression: {}'.format(meanLossRpnR))
                        self.logger.info('Loss Detector classifier: {}'.format(meanLossClassC))
                        self.logger.info('Loss Detector regression: {}'.format(meanLossClassR))
                        self.logger.info('Elapsed time: {}'.format(time.time() - startTime))

                        currentLoss = meanLossRpnC + meanLossRpnR + meanLossClassC + meanLossClassR
                       
                        # If accuracy got better, save weigths
                        if currentLoss < bestLoss:
                            self.logger.info('Total loss decreased from {} to {}, saving weights'.format(bestLoss, currentLoss))
                            bestLoss = currentLoss
                            self.saveModel(neuralNetwork, context)

                        # Finish epoch
                        epochIteration = 0
                        break

                except Exception as e:
                    self.logger.error(e, traceback.format_exc())
                    continue

    def detect(self, context):
        try:
            self.logger.info('Starting detection')
            numROIs = int(context.getConfig('NeuralNetwork', 'num_rois'))
            classMapping = ast.literal_eval(context.getConfig('Training', 'class_mapping'))
            classMapping = {v: k for k, v in classMapping.items()}
            classifier_regr_std = json.loads(context.getConfig('Preprocessing', 'classifier_regr_std'))
            rpn_stride = int(context.getConfig('NeuralNetwork', 'rpn_stride'))
            resize_factor = float(context.getConfig('Preprocessing', 'resize_factor'))
            suppresion_overlap = float(context.getConfig('NeuralNetwork', 'detect_suppresion_overlap'))
            classifier_threshold = float(context.getConfig('NeuralNetwork', 'detect_classifier_threshold'))
            context.classCount = classMapping
            #Initialize loss class for calculations
            frcnnLoss = losses.frcnn.FasterRCNNLoss(1)

            # Create Neural network
            # Initialize network in test mode (isTraining = False)
            neuralNetwork = self.importNetwork(context, frcnnLoss, False)

            # Get preprocessor for image operations
            preprocessor = processors.optimizedPreprocessor.OptimizedImagePreprocessor(self.logger, neuralNetwork)



            self.logger.info('Classes" {}'.format(classMapping))
            #numROIs = int(context.getConfig('NeuralNetwork', 'num_rois'))
            image = preprocessor.loadImage(context.dataPath, context)

            # Process image regions
            self.logger.info('Predicting regions')
            image.predictedRegions = neuralNetwork.predictRPN(image, False)

            self.logger.info('Processing regions')
            regions, probs = preprocessor.rpnToROI(image, context)

            regions, probs = preprocessor.nonMaxSuppression(regions, probs, context)
            self.logger.info('Supressed regions: {}'.format(len(regions)))

            # Convert from (x1,y1,x2,y2) to (x,y,w,h)
            regions[:, 2] -= regions[:, 0] # w = x2 - x1
            regions[:, 3] -= regions[:, 1] # w = y2 - y1

            detectedBoxes = {}
            detectedProbs = {}
            chunks = regions.shape[0]//numROIs

            self.logger.info('Classifying regions')
            for idx in range(chunks + 1):
                ROIs = np.expand_dims(regions[numROIs*idx:numROIs*(idx+1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break

                if idx == chunks:
                    #pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0],numROIs,curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded

                print('Classify ROIs')
                # Feature map = image.predictedRegions[2]
                # Region coords = ROIs
                [P_cls, P_regr] = neuralNetwork.predictClassifier([image.predictedRegions[2], ROIs], False)

                for ii in range(P_cls.shape[1]):
                    if np.max(P_cls[0, ii, :]) < classifier_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                        #print('not enough: ', np.max(P_cls[0, ii, :]))
                        #print('class: ', np.argmax(P_cls[0, ii, :]))
                        continue

                    cls_name = classMapping[np.argmax(P_cls[0, ii, :])]

                    if cls_name not in detectedBoxes:
                        detectedBoxes[cls_name] = []
                        detectedProbs[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]

                    cls_num = np.argmax(P_cls[0, ii, :])
                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                        tx /= classifier_regr_std[0]
                        ty /= classifier_regr_std[1]
                        tw /= classifier_regr_std[2]
                        th /= classifier_regr_std[3]
                        x, y, w, h = preprocessor.apply_regr(x, y, w, h, tx, ty, tw, th)
                    except:
                        pass
                    roi_x1 = int(round(rpn_stride*x))
                    roi_y1 = int(round(rpn_stride*y))
                    roi_x2 = int(round(rpn_stride*(x+w)))
                    roi_y2 = int(round(rpn_stride*(y+h)))
                    detectedBoxes[cls_name].append([roi_x1, roi_y1, roi_x2, roi_y2])
                    detectedProbs[cls_name].append(np.max(P_cls[0, ii, :]))

            finalBoxes = {}
            finalProbs = {}
            for key in detectedBoxes:
                bbox = np.array(detectedBoxes[key])
                probs = np.array(detectedProbs[key])
                print('Suppressing')
                #new_boxes, new_probs = preprocessor.nonMaxSuppression(bbox, probs, context, overlap=suppresion_overlap)
                new_boxes, new_probs = preprocessor.nonMaxSuppressionEuclidean(bbox, probs, context)
                
                x1 = np.expand_dims(np.round(new_boxes[:,0] / resize_factor).astype(np.int),axis=1)
                y1 = np.expand_dims(np.round(new_boxes[:,1] / resize_factor).astype(np.int),axis=1)
                x2 = np.expand_dims(np.round(new_boxes[:,2] / resize_factor).astype(np.int),axis=1)
                y2 = np.expand_dims(np.round(new_boxes[:,3] / resize_factor).astype(np.int),axis=1)

                boxes = np.concatenate((x1,y1,x2,y2), axis=1).tolist()

                # Draw result
                for box in boxes:
                    cv2.rectangle(image.imageData,(box[0], box[1]), (box[2], box[3]), (0,255,0), 1)

                finalResults[key] = []
                for i in range(len(boxes)):
                    detection = {'coords': boxes[i], 'prob': new_probs[i] }
                    finalResults[key].append(detection)

            # Save results
            resultFile = 'result_' + image.id + '.pickle'
            resultImage = 'result_' + image.id + '.' + image.ext
            cv2.imshow('img', image.imageData)
            cv2.waitKey(0)
            cv2.imwrite(resultImage, image.imageData)
            pickle.dump(finalResults, open(resultFile, 'wb'))

        except Exception as e:
            self.logger.error(e, traceback.format_exc())

    def selectROISamples(self, pos_samples, neg_samples, numROIs, remainingROIs):
        try:

            if len(pos_samples) < numROIs//2:
                selected_pos_samples = pos_samples.tolist()
                self.logger.info('Positive samples selected: {}'.format(len(selected_pos_samples)))
            else:
                selected_pos_samples = np.random.choice(pos_samples, numROIs//2, replace=False).tolist()
                self.logger.info('Positive samples selected: {}'.format(len(selected_pos_samples)))
            try:
                selected_neg_samples = np.random.choice(neg_samples, numROIs - len(selected_pos_samples), replace=False).tolist()
                self.logger.info('Negative samples selected: {}'.format(len(selected_neg_samples)))
            except:
                selected_neg_samples = np.random.choice(neg_samples, numROIs - len(selected_pos_samples), replace=True).tolist()
                self.logger.info('Negative samples selected: {}'.format(len(selected_neg_samples)))

            # Remove selected samples from remaining ROIs
            samples_to_remove = selected_neg_samples + selected_pos_samples
            remainingROIs[samples_to_remove] = False

        except Exception as e:
            self.logger.error(e, traceback.format_exc())

        return selected_pos_samples, selected_neg_samples, remainingROIs

    def importNetwork(self, context, loss, isTraining):
        '''Import network gracefully according to architecture name in configuration'''
        neuralNetwork = None
        try:
            architecture = context.getConfig('NeuralNetwork', 'architecture')
            self.logger.info('Creating network: {}'.format(architecture))

            module = importlib.import_module(architecture)
            moduleClasses = inspect.getmembers(module, inspect.isclass)
            # Find network class
            for className, classObject in moduleClasses: 
                if base.NeuralNetwork in classObject.__bases__:
                    networkClassName = className
                    break 
            neuralNetwork = getattr(module, networkClassName)(context, loss, isTraining)
        except Exception as e:
            self.logger.error('Could not import network module, falling back to default (Resnet50)', traceback.format_exc())
            import resnet50
            neuralNetwork = resnet50.ResNet50Network(context, loss, isTraining)
        
        return neuralNetwork

    def saveModel(self, neuralNetwork, context):
        # TODO: Save with timestamp
        try:
            architecture = context.getConfig('NeuralNetwork', 'architecture')
            stride = context.getConfig('NeuralNetwork', 'rpn_stride')
            resize = context.getConfig('Preprocessing', 'resize_factor')
            anchor_sizes = json.loads(context.getConfig('NeuralNetwork', 'anchor_scales'))
            anchor_ratios = json.loads(context.getConfig('NeuralNetwork', 'anchor_ratios'))
            num_anchors = len(anchor_sizes) * len(anchor_ratios)	
            output = context.getConfig('NeuralNetwork', 'output_path')
            filename = 'model_' + architecture + '_n' + str(num_anchors) + '_s' + stride + '_r' + resize + '.hdf5'

            if not os.path.isdir(os.path.abspath(output)):
                os.mkdir(os.path.abspath(output))

            filepath = os.path.join(os.path.abspath(output), filename)

            neuralNetwork.saveWeights(filepath)
        except Exception as e:
            self.logger.error(e, traceback.format_exc())

    #endregion


