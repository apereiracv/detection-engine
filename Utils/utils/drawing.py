import cv2
import numpy as np

import entities

class ImageDrawer(object):

    @staticmethod
    def drawRegionsOnImage(image, boxes, context, probs = None, unique_id = ''):
        stride = int(context.getConfig('NeuralNetwork', 'rpn_stride'))
        resize = context.getConfig('Preprocessing', 'resize_factor')
        filename = image.id + '_s' + str(stride) + '_r' + resize + unique_id + '.jpg'
        img = image.imageData.astype(int)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        # Transform regions to original size
        x1 = (x1 * stride / float(resize)).astype(int)
        y1 = (y1 * stride / float(resize)).astype(int)
        x2 = (x2 * stride / float(resize)).astype(int)
        y2 = (y2 * stride / float(resize)).astype(int)

        (numberOfRegions, _) = boxes.shape
        for box in range(numberOfRegions):
            # Change color intensity according to objectness probability
            if probs is None:
                green = 255
            else:
                green = int(255 * probs[box])

            color = (0,green,0)

            img = cv2.rectangle(img,(x1[box],y1[box]),(x2[box],y2[box]),color,1)

        cv2.imwrite(filename,img)

    @staticmethod
    def drawRegionsOnImageWH(image, rois, context, roiClassification = None, unique_id = ''):
        stride = int(context.getConfig('NeuralNetwork', 'rpn_stride'))
        resize = context.getConfig('Preprocessing', 'resize_factor')
        filename = image.id + '_s' + str(stride) + '_r' + resize + unique_id + 'WH.jpg'
        img = image.imageData.astype(int)
        (_, roiNumber, _) = rois.shape
        for roi in range(roiNumber):
            x1 = int(rois[0][roi][0] * stride / float(resize))
            y1 = int(rois[0][roi][1] * stride / float(resize))
            x2 = int((rois[0][roi][0] + rois[0][roi][2]) * stride / float(resize))
            y2 = int((rois[0][roi][1] + rois[0][roi][3]) * stride / float(resize))

            color = (0,255,0)

            if (roiClassification is not None) and (roiClassification[0, roi, -1] == 1):
                color = (0,0,255)
                #img = cv2.rectangle(img,(x1,y1),(x2,y2),color,1)
            else:
                img = cv2.rectangle(img,(x1,y1),(x2,y2),color,1)

        cv2.imwrite(filename,img)

    @staticmethod
    def drawImage(image):
        filename = image.id + '_augmented' + '.' + image.ext
        img = image.imageData.astype(int)

        for box in image.boundingBoxes:
            color = (0,255,0) # green
            img = cv2.rectangle(img,(box['x1'],box['y1']),(box['x2'],box['y2']),color,1)

        cv2.imwrite(filename,img)
