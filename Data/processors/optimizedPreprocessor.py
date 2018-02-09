import cv2
import numpy as np
import multiprocessing
import json
import random
import copy
import traceback
import os
import pickle
import math
import ntpath
import utils.drawing
import ast

import entities.image

class OptimizedImagePreprocessor(object):
    
    def __init__(self, logger, neuralNetwork):
        self.logger = logger
        self.getFeatureMapSize = neuralNetwork.getFeatureMapSize

    #region Overlap calculations
    def apply_regr(self, x, y, w, h, tx, ty, tw, th):
        try:
            cx = x + w/2.
            cy = y + h/2.
            cx1 = tx * w + cx
            cy1 = ty * h + cy
            w1 = math.exp(tw) * w
            h1 = math.exp(th) * h
            x1 = cx1 - w1/2.
            y1 = cy1 - h1/2.
            #x1 = int(round(x1))
            #y1 = int(round(y1))
            #w1 = int(round(w1))
            #h1 = int(round(h1))
        
            if w1 == 0:
                w1 = 1

            if h1 == 0:
                h1 = 1

            return x1, y1, w1, h1

        except ValueError:
            return x, y, w, h
        except OverflowError:
            return x, y, w, h
        except Exception as e:
            self.logger.error(e, traceback.format_exc())
            return x, y, w, h

    def apply_regr_np(self, X, T):
        try:
            x = X[0, :, :]
            y = X[1, :, :]
            w = X[2, :, :]
            h = X[3, :, :]

            tx = T[0, :, :]
            ty = T[1, :, :]
            tw = T[2, :, :]
            th = T[3, :, :]

            cx = x + w/2.
            cy = y + h/2.
            cx1 = tx * w + cx
            cy1 = ty * h + cy

            tw = tw.astype(np.float64)
            th = th.astype(np.float64)

            w1 = np.exp(tw) * w
            h1 = np.exp(th) * h
            x1 = cx1 - w1/2.
            y1 = cy1 - h1/2.

            #x1 = np.round(x1)
            #y1 = np.round(y1)
            #w1 = np.round(w1)
            #h1 = np.round(h1)
            return np.stack([x1, y1, w1, h1])
        except Exception as e:
            self.logger.error(e, traceback.format_exc())
            return X
    #endregion

    #region Image operations
    def resizeImage(self, image, resizeFactor):
        resizedWidth = int(resizeFactor * image.width)
        resizedHeight = int( resizeFactor * image.height)
        resizedImageData = cv2.resize(image.imageData, (resizedWidth, resizedHeight), interpolation=cv2.INTER_CUBIC)

        return resizedImageData, resizedWidth, resizedHeight


    def normalizeImageData(self, imageData, context):
        # Zero-center by mean pixel and re-arrange arrays
        img_channels_mean = json.loads(context.getConfig('Preprocessing', 'image_channels_mean'))
        img_scaling_factor = float(context.getConfig('Preprocessing', 'image_scaling_factor'))

        #x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
        x_img = imageData.astype(np.float32)
        x_img[:, :, 0] -= img_channels_mean[0]
        x_img[:, :, 1] -= img_channels_mean[1]
        x_img[:, :, 2] -= img_channels_mean[2]
        x_img /= img_scaling_factor

        #TODO: revisit, are 2 transposed needed?
        x_img = np.transpose(x_img, (2, 0, 1))
        x_img = np.expand_dims(x_img, axis=0)

        # Transposing needed for TF
        x_img = np.transpose(x_img, (0, 2, 3, 1))

        return x_img


    def augmentImage(self, image, context):
        rotate = context.getBoolean('Training', 'use_rotations')
        flip = context.getBoolean('Training', 'use_flips')

        image.originalImageData = copy.deepcopy(image.imageData)
        img = image.imageData
        rows, cols = img.shape[:2]

        h_flip = False
        v_flip = False
        rot = 0
        # Horizontal flip
        if flip and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)
            for bbox in image.boundingBoxes:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = cols - x1
                bbox['x1'] = cols - x2

            h_flip = True

        # Vertical flip
        if flip and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 0)
            for bbox in image.boundingBoxes:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = rows - y1
                bbox['y1'] = rows - y2

            v_flip = True

        # Rotation
        if rotate:
            angle = np.random.choice([0,90,180,270],1)[0]
            if angle == 270:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img, 0)
            elif angle == 180:
                img = cv2.flip(img, -1)
            elif angle == 90:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img, 1)
            elif angle == 0:
                pass
            rot = angle
            for bbox in image.boundingBoxes:
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']
                if angle == 270:
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = cols - x2
                    bbox['y2'] = cols - x1
                elif angle == 180:
                    bbox['x2'] = cols - x1
                    bbox['x1'] = cols - x2
                    bbox['y2'] = rows - y1
                    bbox['y1'] = rows - y2
                elif angle == 90:
                    bbox['x1'] = rows - y2
                    bbox['x2'] = rows - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2        
                elif angle == 0:
                    pass
            

        # Set final image data
        image.imageData = img
        image.width =  img.shape[1]
        image.height = img.shape[0]
        image.horizontalFlip = h_flip
        image.verticalFlip = v_flip
        image.rotation = rot

        return image


    def getImageDataIterator(self, imageList, context):
        while True:
            # TODO: Randomize
            for image in imageList:
                # Clean up previous data
                image.cleanUp()
                self.logger.info('Loading image: {}'.format(image.filePath))
                image.imageData = cv2.imread(image.filePath)
                (image.height, image.width, _) = image.imageData.shape
                # Use rotations and flips if activated
                image = self.augmentImage(image, context)
                #utils.drawing.ImageDrawer.drawImage(image) # Draw image to test augment functionality
                resizeFactor = float(context.getConfig('Preprocessing', 'resize_factor'))
                # Resize to scaling factor and normalize pixels range
                image.resizedImageData, image.resizedWidth, image.resizedHeight = self.resizeImage(image, resizeFactor) 
                image.resizedImageData = self.normalizeImageData(image.resizedImageData, context)

                # Calculate RPN regions for training objective
                try:
                    self.logger.info('Getting RPN regions')
                    y_rpn_cls, y_rpn_regr = self.getRegions(image, context)                    

                    std_scaling = float(context.getConfig('Preprocessing', 'std_scaling'))
                    y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= std_scaling

                    # Transposing needed for TF
                    y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                    y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                    # Save calculated regions into image object
                    image.trainingRegions = [y_rpn_cls, y_rpn_regr]

                    yield image

                except Exception as e:
                    self.logger.error(e, traceback.format_exc())
                    continue
       
     
    def getRegions(self, image, context):
        # Region file name according to generated regions settings ex: surface_top_s16_r2
        stride = context.getConfig('NeuralNetwork', 'rpn_stride')
        resize = context.getConfig('Preprocessing', 'resize_factor')
        filename = image.id +\
                   '_s' + stride +\
                   '_r' + resize +\
                   '_hf' + str(int(image.horizontalFlip)) +\
                   '_vf' + str(int(image.verticalFlip)) +\
                   '_rot' + str(image.rotation) + '.pickle'
        
        regionDir = context.getConfig('NeuralNetwork', 'regions_path')
        if not os.path.isdir(os.path.abspath(regionDir)):
            os.mkdir(os.path.abspath(regionDir))

        imageRegionFile = os.path.join(os.path.abspath(regionDir), filename)

        if os.path.isfile(imageRegionFile):
            self.logger.info('Loading regions from file')
            regions = pickle.load(open(imageRegionFile, "rb"))
            y_rpn_cls = regions[0]
            y_rpn_regr = regions[1]
        else:
            self.logger.info('Calculating regions')
            y_rpn_cls, y_rpn_regr = self.calculateRPN(image, context)   
            regions = [y_rpn_cls, y_rpn_regr]
            pickle.dump(regions, open(imageRegionFile, "wb"))

        return y_rpn_cls, y_rpn_regr


    def calculateRPN(self, image, context):
        # Context settings
        rpn_stride = float(context.getConfig('NeuralNetwork', 'rpn_stride'))
        anchor_sizes = json.loads(context.getConfig('NeuralNetwork', 'anchor_scales'))
        anchor_ratios = json.loads(context.getConfig('NeuralNetwork', 'anchor_ratios'))
        rpn_min_overlap = float(context.getConfig('NeuralNetwork','rpn_min_overlap'))
        rpn_max_overlap = float(context.getConfig('NeuralNetwork','rpn_max_overlap'))
        resize_factor = float(context.getConfig('Preprocessing', 'resize_factor'))
        num_anchors = len(anchor_sizes) * len(anchor_ratios)	
        num_bboxes = len(image.boundingBoxes)

        # Calculate the output feature map size based on the network architecture
        featureMapWidth, featureMapHeight = self.getFeatureMapSize(image.resizedWidth, image.resizedHeight)
    
        # Initialize anchor regions
        anchor_regions_x1 = np.zeros((featureMapHeight, featureMapWidth, num_anchors))
        anchor_regions_y1 = np.zeros((featureMapHeight, featureMapWidth, num_anchors))
        anchor_regions_x2 = np.zeros((featureMapHeight, featureMapWidth, num_anchors))
        anchor_regions_y2 = np.zeros((featureMapHeight, featureMapWidth, num_anchors))

        # Create anchor list from all combinations between sizes and ratios
        anchor_list = self.getAnchorList(anchor_sizes, anchor_ratios, resize_factor)

        # Calculate anchor coordinates from feature maps
        for ix in range(featureMapWidth):
            for jy in range(featureMapHeight):
                for num_anchor in range(len(anchor_list)): 
                    anchor_x = anchor_list[num_anchor][0]
                    anchor_y = anchor_list[num_anchor][1]
                    anchor_regions_x1[jy,ix,num_anchor] = rpn_stride * (ix + 0.5) - anchor_x / 2
                    anchor_regions_y1[jy,ix,num_anchor] = rpn_stride * (jy + 0.5) - anchor_y / 2
                    anchor_regions_x2[jy,ix,num_anchor] = rpn_stride * (ix + 0.5) + anchor_x / 2
                    anchor_regions_y2[jy,ix,num_anchor] = rpn_stride * (jy + 0.5) + anchor_y / 2

        # Get only valid regions inside the image boundaries
        valid_regions = np.where(np.logical_and.reduce((anchor_regions_x1[:, :, :].flatten() > 0, 
                                                        anchor_regions_y1[:, :, :].flatten() > 0,
                                                        anchor_regions_x2[:, :, :].flatten() <= image.resizedWidth,
                                                        anchor_regions_y2[:, :, :].flatten() <= image.resizedHeight)))[0]

        # Calculate IOU over all valid regions
        anchor_x1_valid = anchor_regions_x1.flatten()[valid_regions]
        anchor_y1_valid = anchor_regions_y1.flatten()[valid_regions]
        anchor_x2_valid = anchor_regions_x2.flatten()[valid_regions]
        anchor_y2_valid = anchor_regions_y2.flatten()[valid_regions]

        anchor_regions_valid = np.zeros((len(anchor_x1_valid), 4))
        anchor_regions_valid[:,0] = anchor_x1_valid
        anchor_regions_valid[:,1] = anchor_y1_valid
        anchor_regions_valid[:,2] = anchor_x2_valid
        anchor_regions_valid[:,3] = anchor_y2_valid

        # Get ground truth array from bouding boxes, resizing according to resize factor
        gta = self.calculateGroundTruth(image)
    
        overlaps = self.calculateArrayIOU(anchor_regions_valid, gta)

        tx, ty, tw, th = self.calculateArrayRegr(anchor_regions_valid, gta, True)

        # initialize empty output objectives
        y_rpn_overlap = np.zeros((featureMapHeight, featureMapWidth, num_anchors)).flatten()
        y_is_box_valid = np.zeros((featureMapHeight, featureMapWidth, num_anchors)).flatten()
        y_rpn_regr = np.zeros((featureMapHeight, featureMapWidth, num_anchors * 4)).flatten()

        # Select only overlaps that are over the threshold or are the best ones for a specific bbox
        pos_overlaps_max = np.where(overlaps >= rpn_max_overlap)
        pos_overlaps_bbox = (np.arange(num_bboxes), overlaps.argmax(axis=1))
        neg_overlaps = overlaps.max(axis=0)
        neg_overlaps = np.where(neg_overlaps <= rpn_min_overlap)[0]

        # Activate all regions with overlap > max
        if len(pos_overlaps_max[1]) > 0:
            y_is_box_valid[valid_regions[pos_overlaps_max[1]]] = 1
            y_rpn_overlap[valid_regions[pos_overlaps_max[1]]] = 1
            
            start = 4 * valid_regions[pos_overlaps_max[1]]
            y_rpn_regr[start] = np.copy(tx[pos_overlaps_max])
            y_rpn_regr[start + 1] = np.copy(ty[pos_overlaps_max])
            y_rpn_regr[start + 2] = np.copy(tw[pos_overlaps_max])
            y_rpn_regr[start + 3] = np.copy(th[pos_overlaps_max])

        # Activate all regions that are the best for a bbox
        if len(pos_overlaps_bbox[1]) > 0:
            y_is_box_valid[valid_regions[pos_overlaps_bbox[1]]] = 1
            y_rpn_overlap[valid_regions[pos_overlaps_bbox[1]]] = 1

            start = 4 * valid_regions[pos_overlaps_bbox[1]]
            y_rpn_regr[start] = np.copy(tx[pos_overlaps_bbox])
            y_rpn_regr[start + 1] = np.copy(ty[pos_overlaps_bbox])
            y_rpn_regr[start + 2] = np.copy(tw[pos_overlaps_bbox])
            y_rpn_regr[start + 3] = np.copy(th[pos_overlaps_bbox])

        # Activate negative regions
        y_is_box_valid[valid_regions[np.unique(neg_overlaps)]] = 1

        del tx,ty, tw, th
        del pos_overlaps_bbox, pos_overlaps_max, neg_overlaps

        # Reshape to contine calculations
        y_rpn_overlap = y_rpn_overlap.reshape((featureMapHeight, featureMapWidth, num_anchors))
        y_is_box_valid = y_is_box_valid.reshape((featureMapHeight, featureMapWidth, num_anchors))
        y_rpn_regr = y_rpn_regr.reshape((featureMapHeight, featureMapWidth, num_anchors * 4))

        y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
        y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

        y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
        y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

        y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
        y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

        pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
        neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))
    
        # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
        # regions. We also limit it to 2048 regions.
        num_regions = 2048
        num_pos = len(pos_locs[0])

        if len(pos_locs[0]) > int(num_regions/2):
            val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - int(num_regions/2))
            y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
            num_pos = int(num_regions/2)

        if len(neg_locs[0]) + num_pos > num_regions:
            val_locs = random.sample(range(len(neg_locs[0])), int(len(neg_locs[0]) - num_pos))
            y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

        y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
        y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

        return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


    def rpnToROI(self, image, context):
        anchor_sizes = json.loads(context.getConfig('NeuralNetwork', 'anchor_scales'))
        anchor_ratios = json.loads(context.getConfig('NeuralNetwork', 'anchor_ratios'))
        rpn_stride = int(context.getConfig('NeuralNetwork', 'rpn_stride'))
        std_scaling = float(context.getConfig('Preprocessing', 'std_scaling'))
        resize_factor = float(context.getConfig('Preprocessing', 'resize_factor'))
        rpn_layer = image.predictedRegions[0]
        regr_layer = image.predictedRegions[1] / std_scaling
        
        assert rpn_layer.shape[0] == 1

        (rows, cols) = rpn_layer.shape[1:3]

        curr_layer = 0
        A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))
        
        for anchor_size in anchor_sizes:
            for anchor_ratio in anchor_ratios:

                anchor_x = (anchor_size * anchor_ratio[0]) / rpn_stride * resize_factor
                anchor_y = (anchor_size * anchor_ratio[1]) / rpn_stride * resize_factor

                regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
                regr = np.transpose(regr, (2, 0, 1))

                X, Y = np.meshgrid(np.arange(cols),np.arange(rows))

                A[0, :, :, curr_layer] = X - anchor_x/2
                A[1, :, :, curr_layer] = Y - anchor_y/2
                A[2, :, :, curr_layer] = anchor_x
                A[3, :, :, curr_layer] = anchor_y

                A[:, :, :, curr_layer] = self.apply_regr_np(A[:, :, :, curr_layer], regr)

                A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
                A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
                A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
                A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

                A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
                A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
                A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer])
                A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

                curr_layer += 1

        all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
        all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

        x1 = all_boxes[:, 0]
        y1 = all_boxes[:, 1]
        x2 = all_boxes[:, 2]
        y2 = all_boxes[:, 3]

        idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

        all_boxes = np.delete(all_boxes, idxs, 0)
        all_probs = np.delete(all_probs, idxs, 0)

        return all_boxes, all_probs


    def nonMaxSuppression(self, boxes, probs, context, overlap=None):
        # code example from: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        if overlap is None:
            overlap_thresh = float(context.getConfig('NeuralNetwork', 'suppresion_overlap'))
        else:
            overlap_thresh = overlap

        max_boxes = int(context.getConfig('NeuralNetwork', 'max_boxes'))

        # Sort bounding boxes by probabilities
        idxs = np.argsort(probs)
        boxes = boxes[idxs]
        probs = probs[idxs]

        self.logger.info('Total proposed regions: {}'.format(len(boxes)))

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        np.testing.assert_array_less(x1, x2)
        np.testing.assert_array_less(y1, y2)

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # calculate the areas
        area = (x2 - x1) * (y2 - y1)
    
        # Create constants dict for processes
        constants = {
            'overlap_treshold' : overlap_thresh,
            'area' : area,
            'axes' : (x1, y1, x2, y2),
            'max_boxes': max_boxes
        }

        # Create shared variables between processes
        shared_last_index = multiprocessing.Value('i', (len(idxs) - 1))

        deleted_items = np.ones(len(idxs), dtype=np.bool).tolist()
        picked_items = np.zeros(len(idxs), dtype=np.bool).tolist()

        shared_deleted_items = multiprocessing.Array('b', deleted_items)
        shared_picked_items = multiprocessing.Array('b', picked_items)

        lock = multiprocessing.Lock()

        num_processes = multiprocessing.cpu_count()
        processes_array = []

        for process_num in range(num_processes):
            nms_process = multiprocessing.context.Process(target = nonMaxSuppresionProcess, args = (constants, shared_picked_items, shared_deleted_items, shared_last_index, lock, process_num))
            processes_array.append(nms_process)

        # Start all processes
        for process in processes_array:
            process.start()

        # Wait for all processes to finish
        for process in processes_array:
            process.join()

        # Return only the picked boxes from the indexes
        final_picked_indexes = np.copy(np.ctypeslib.as_array(shared_picked_items.get_obj())).astype(np.bool)

        resultBoxes = boxes[final_picked_indexes].astype("int")
        resultProbs = probs[final_picked_indexes]

        return resultBoxes, resultProbs


    def nonMaxSuppressionWithIOU(self, image, boxes, probs, context):
        rpn_stride = int(context.getConfig('NeuralNetwork', 'rpn_stride'))
        suppresion_overlap = float(context.getConfig('NeuralNetwork', 'suppresion_overlap'))
        classifier_min_overlap = float(context.getConfig('NeuralNetwork', 'classifier_min_overlap'))
        classifier_max_overlap = float(context.getConfig('NeuralNetwork', 'classifier_max_overlap'))
        classifier_regr_std = json.loads(context.getConfig('Preprocessing', 'classifier_regr_std'))
        class_mapping = ast.literal_eval(context.getConfig('Training', 'class_mapping'))

        bboxes = image.boundingBoxes
        num_bboxes = len(bboxes)
        (width, height) = (image.width, image.height)
        (resized_width, resized_height) = (image.resizedWidth, image.resizedHeight)

        gta = self.calculateGroundTruth(image, rpn_stride)
        overlaps = self.calculateArrayIOU(boxes, gta)
        
        # Get valid overlaps: Regions over the minimum
        valid_overlaps = np.where(overlaps > classifier_min_overlap)
        del overlaps

        # Order boxes by descending order of objectness probability
        valid_probs = probs[valid_overlaps[1]]
        valid_boxes = boxes[valid_overlaps[1]]
        ordered_idxs = np.argsort(valid_probs)[::-1]
        valid_boxes = valid_boxes[ordered_idxs]
        valid_probs = valid_probs[ordered_idxs]
        del valid_overlaps

        overlaps_suppresion = self.calculateArrayIOU(valid_boxes, valid_boxes)
        full_overlap = 0.98 #TODO: Make constant
        suppresed_overlaps = np.logical_and(np.tril(overlaps_suppresion) > suppresion_overlap, np.tril(overlaps_suppresion) < full_overlap)
        full_overlaps = np.where(overlaps_suppresion >= full_overlap)
        overlaps_suppresion[suppresed_overlaps] = 1
        overlaps_suppresion[full_overlaps] = 1
        suppresed_idxs = []
        max_overlap = overlaps_suppresion.argmax(axis=1)

        for idx in range(len(valid_boxes)):
            if max_overlap[idx] == idx:
                suppresed_idxs.append(idx)

        suppresed_boxes = valid_boxes[suppresed_idxs]
        suppresed_probs = valid_probs[suppresed_idxs]
        del overlaps_suppresion, suppresed_overlaps, full_overlaps, max_overlap

        final_overlaps = self.calculateArrayIOU(suppresed_boxes, gta)
        pos_overlaps = np.where(final_overlaps >= classifier_max_overlap)
        # TODO: Check why some boxes neg are also pos

        pos_boxes = suppresed_boxes[pos_overlaps[1]]

        # Initialize empty objective arrays
        x_roi = np.zeros((len(suppresed_boxes), 4))
        y_class_num = np.zeros((len(suppresed_boxes),len(class_mapping)))
        y_class_regr_coords = np.zeros((len(suppresed_boxes), len(class_mapping) - 1, 4))
        y_class_regr_label = np.zeros((len(suppresed_boxes), len(class_mapping) - 1, 4))

        x1 = suppresed_boxes[:, 0]
        y1 = suppresed_boxes[:, 1]
        x2 = suppresed_boxes[:, 2]
        y2 = suppresed_boxes[:, 3]

        # Fill up objective arrays
        x_roi[:,0] = x1
        x_roi[:,1] = y1
        x_roi[:,2] = (x2 - x1)
        x_roi[:,3] = (y2 - y1)

        tx, ty, tw, th = self.calculateArrayRegr(pos_boxes, gta, False)
        sx, sy, sw, sh = classifier_regr_std
        # Set class enconding for bg
        bg_class_num = class_mapping['bg']
        y_class_num[:,bg_class_num] = 1

        for idx in range(len(pos_overlaps[0])):
            region_idx = pos_overlaps[1][idx]
            bbox_idx = pos_overlaps[0][idx]
            class_num = class_mapping[bboxes[bbox_idx]['class']]
            y_class_num[region_idx][class_num] = 1
            y_class_num[region_idx][bg_class_num] = 0
            y_class_regr_coords[region_idx][class_num][0] = sx * tx[bbox_idx][idx]
            y_class_regr_coords[region_idx][class_num][1] = sy * ty[bbox_idx][idx]
            y_class_regr_coords[region_idx][class_num][2] = sw * tw[bbox_idx][idx]
            y_class_regr_coords[region_idx][class_num][3] = sh * th[bbox_idx][idx]
            y_class_regr_label[region_idx][class_num] = [1,1,1,1]
        
        
        y_class_regr_label = y_class_regr_label.reshape((len(suppresed_boxes), (len(class_mapping) - 1) * 4))
        y_class_regr_coords = y_class_regr_coords.reshape((len(suppresed_boxes), (len(class_mapping) - 1) * 4))

        X = np.expand_dims(x_roi, axis=0)
        Y1 = np.expand_dims(y_class_num, axis=0)
        Y2 = np.expand_dims(np.concatenate([y_class_regr_label, y_class_regr_coords],axis=1), axis=0)

        return X, Y1, Y2


    def calculateArrayIOU(self, arrayA, arrayB):
        # Calculate all regions IOUs with numpy operations
        # Get coords again only for valid regions
        x1_a = arrayA[:, 0]
        y1_a = arrayA[:, 1]
        x2_a = arrayA[:, 2]
        y2_a = arrayA[:, 3]
        area_a = (x2_a - x1_a) * (y2_a - y1_a)

        x1_b = arrayB[:, 0]
        y1_b = arrayB[:, 1]
        x2_b = arrayB[:, 2]
        y2_b = arrayB[:, 3]
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        

        xx1_int = np.maximum(x1_a, np.expand_dims(x1_b,axis=1))
        xx2_int = np.minimum(x2_a, np.expand_dims(x2_b,axis=1))
        ww_int = np.maximum(0, xx2_int - xx1_int)
        del xx1_int, xx2_int

        yy1_int = np.maximum(y1_a, np.expand_dims(y1_b,axis=1))
        yy2_int = np.minimum(y2_a, np.expand_dims(y2_b,axis=1))
        hh_int = np.maximum(0, yy2_int - yy1_int)
        del yy1_int, yy2_int

        area_int = np.copy(ww_int * hh_int)
        del ww_int, hh_int

        # find the union
        area_union = (area_a + np.expand_dims(area_b,axis=1)) - area_int


        # Compute overlaps
        overlaps = np.copy(area_int/(area_union + 1e-6))
        del area_int, area_union

        return overlaps


    def calculateArrayRegr(self, arrayA, arrayB, isRPN):
        x1_a = arrayA[:, 0]
        y1_a = arrayA[:, 1]
        x2_a = arrayA[:, 2]
        y2_a = arrayA[:, 3]

        x1_b = arrayB[:, 0]
        y1_b = arrayB[:, 1]
        x2_b = arrayB[:, 2]
        y2_b = arrayB[:, 3]

        w = x2_a - x1_a
        h = y2_a - y1_a

        if isRPN:
            cx_a = (x1_a + x2_a) /2.0
            cy_a = (y1_a + y2_a) /2.0
        else:
            cx_a = x1_a + w /2.0
            cy_a = y1_a + h /2.0

        cx_b = np.expand_dims(((x1_b + x2_b) / 2.0), axis=1)
        cy_b = np.expand_dims(((y1_b + y2_b) / 2.0), axis=1)

        tx1 = (cx_b - cx_a)
        tx = tx1 / w
        ty1 = (cy_b - cy_a)
        ty = ty1 / h
        del cx_a, cy_a
        del cx_b, cy_b
        tw = np.log(np.expand_dims((x2_b - x1_b),axis=1) / w)
        th = np.log(np.expand_dims((y2_b - y1_b),axis=1) / h)
        del tx1, w
        del ty1, h

        return tx, ty, tw, th


    def calculateGroundTruth(self, image, rpn_stride = None):
        # get the GT box coordinates, and resize to account for image resizing
        gta = np.zeros((len(image.boundingBoxes), 4))

        if rpn_stride is None:
            rpn_stride = 1
        for bbox_num, bbox in enumerate(image.boundingBoxes):
            # get the GT box coordinates, and resize to account for image resizing
            # TODO: Check, is width and height correct?
            gta[bbox_num, 0] = bbox['x1'] * (image.resizedWidth / float(image.width) / rpn_stride)
            gta[bbox_num, 1] = bbox['y1'] * (image.resizedWidth / float(image.width) / rpn_stride)
            gta[bbox_num, 2] = bbox['x2'] * (image.resizedHeight / float(image.height) / rpn_stride)
            gta[bbox_num, 3] = bbox['y2'] * (image.resizedHeight / float(image.height) / rpn_stride)

        return gta
    

    # Find the distance in pixels between the center of each bounding box to the rest
    def calculateEuclideanDistance(self, arrayA, arrayB):
        x1_a = arrayA[:, 0]
        y1_a = arrayA[:, 1]
        x2_a = arrayA[:, 2]
        y2_a = arrayA[:, 3]

        x1_b = arrayB[:, 0]
        y1_b = arrayB[:, 1]
        x2_b = arrayB[:, 2]
        y2_b = arrayB[:, 3]

        # Calculate center points of all boxes
        x_center_a = x1_a + (x2_a - x1_a)/2
        y_center_a = y1_a + (y2_a - y1_a)/2
        x_center_b = np.expand_dims(x1_b + (x2_b - x1_b)/2, axis=1)
        y_center_b = np.expand_dims(y1_b + (y2_b - y1_b)/2, axis=1)

        # Euclidean distance = sqrt((bx - ax)^2 + (by - ay)^2)
        x_distance = np.power(x_center_b - x_center_a, 2)
        y_distance = np.power(y_center_b - y_center_a, 2)
        distances = np.sqrt(x_distance + y_distance)

        return distances


    def nonMaxSuppressionEuclidean(self, boxes, probs, context):
        min_center_distance = int(context.getConfig('NeuralNetwork', 'min_center_distance'))

        ordered_idxs = np.argsort(probs)[::-1]
        boxes = boxes[ordered_idxs]
        probs = probs[ordered_idxs]

        distances = self.calculateEuclideanDistance(boxes, boxes)

        suppresed_boxes = np.tril(distances < min_center_distance)
        distances[suppresed_boxes] = 0
        min_distances = distances.argmin(axis=1)
        selected_idxs = []

        for idx in range(len(boxes)):
            if min_distances[idx] == idx:
                selected_idxs.append(idx)
        

        result_boxes = boxes[selected_idxs]
        result_probs = probs[selected_idxs]

        return result_boxes, result_probs


    def getAnchorList(self, anchor_sizes, anchor_ratios, resize_factor = 1.0):
        # Create anchor list from all combinations between sizes and ratios
        anchor_list = []
        for size_idx in range(len(anchor_sizes)):
            for ratio_idx in range(len(anchor_ratios)):
                anchor_x = anchor_sizes[size_idx] * anchor_ratios[ratio_idx][0] * resize_factor
                anchor_y = anchor_sizes[size_idx] * anchor_ratios[ratio_idx][1] * resize_factor
                anchor_list.append([anchor_x,anchor_y])

        return anchor_list


    def loadImage(self, imagePath, context):
        newImage = entities.image.Image(imagePath)
        self.logger.info('Loading image: {}'.format(newImage.filePath))
        newImage.imageData = cv2.imread(newImage.filePath)
        (newImage.height, newImage.width, _) = newImage.imageData.shape

        resizeFactor = float(context.getConfig('Preprocessing', 'resize_factor'))
        # Resize to scaling factor and normalize pixels range
        newImage.resizedImageData, newImage.resizedWidth, newImage.resizedHeight = self.resizeImage(newImage, resizeFactor) 
        newImage.resizedImageData = self.normalizeImageData(newImage.resizedImageData, context)  
        filename = ntpath.basename(imagePath)
        newImage.id = filename.split('.')[0]
        newImage.ext = filename.split('.')[1]
        return newImage

    #endregion

#region External functions
def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h

def iou(a, b):
    # a and b should be (x1,y1,x2,y2)
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)

def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union

def calculateRPNProcess(constants, anchor_size_idx, anchor_ratio_idx, shared_y_rpn_overlap, 
                               shared_y_is_box_valid, shared_y_rpn_regr, shared_num_anchors_for_bbox, 
                               shared_best_anchor_for_bbox, shared_best_iou_for_bbox, shared_best_x_for_bbox, shared_best_dx_for_bbox):
    # Extract constant values
    downscale = constants['downscale']
    rpn_min_overlap = constants['rpn_min_overlap']
    rpn_max_overlap = constants['rpn_max_overlap']

    anchor_sizes = constants['anchor_sizes']
    anchor_ratios = constants['anchor_ratios']

    output_width = constants['output_width']
    output_height = constants['output_height']

    resized_width = constants['resized_width']
    resized_height = constants['resized_height']

    n_anchratios = constants['n_anchratios']
    gta = constants['gta']
    boundingBoxes = constants['boundingBoxes']
    num_bboxes = constants['num_bboxes']
    num_anchors = constants['num_anchors']

    # Transform shared arrays to numpy arrays
    y_rpn_overlap = np.ctypeslib.as_array(shared_y_rpn_overlap.get_obj())
    y_rpn_overlap = y_rpn_overlap.reshape((output_height, output_width, num_anchors))

    y_is_box_valid = np.ctypeslib.as_array(shared_y_is_box_valid.get_obj())
    y_is_box_valid = y_is_box_valid.reshape((output_height, output_width, num_anchors))

    y_rpn_regr = np.ctypeslib.as_array(shared_y_rpn_regr.get_obj())
    y_rpn_regr = y_rpn_regr.reshape((output_height, output_width, num_anchors * 4))

    num_anchors_for_bbox = np.ctypeslib.as_array(shared_num_anchors_for_bbox.get_obj())

    best_anchor_for_bbox = np.ctypeslib.as_array(shared_best_anchor_for_bbox.get_obj())
    best_anchor_for_bbox = best_anchor_for_bbox.reshape((num_bboxes, 4))

    best_iou_for_bbox = np.ctypeslib.as_array(shared_best_iou_for_bbox.get_obj())

    best_x_for_bbox = np.ctypeslib.as_array(shared_best_x_for_bbox.get_obj())
    best_x_for_bbox = best_x_for_bbox.reshape((num_bboxes, 4))

    best_dx_for_bbox = np.ctypeslib.as_array(shared_best_dx_for_bbox.get_obj())
    best_dx_for_bbox = best_dx_for_bbox.reshape((num_bboxes, 4))

    # Start calculations
    anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
    anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]
    #print('Thread with anchor: ( {}, {} )'.format(anchor_x, anchor_y))
            
    for ix in range(output_width):	
        # x-coordinates of the current anchor box	
        x1_anc = downscale * (ix + 0.5) - anchor_x / 2
        x2_anc = downscale * (ix + 0.5) + anchor_x / 2	
                
        # ignore boxes that go across image boundaries					
        if x1_anc < 0 or x2_anc > resized_width:
            continue
                    
        for jy in range(output_height):

            # y-coordinates of the current anchor box
            y1_anc = downscale * (jy + 0.5) - anchor_y / 2
            y2_anc = downscale * (jy + 0.5) + anchor_y / 2

            # ignore boxes that go across image boundaries
            if y1_anc < 0 or y2_anc > resized_height:
                continue

            # bbox_type indicates whether an anchor should be a target 
            bbox_type = 'neg'

            # this is the best IOU for the (x,y) coord and the current anchor
            # note that this is different from the best IOU for a GT bbox
            best_iou_for_loc = 0.0

            for bbox_num in range(num_bboxes):
                #print('bbox: {}'.format(bbox_num))	
                # get IOU of the current GT box and the current anchor box
                curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
                # calculate the regression targets if they will be needed
                if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > rpn_max_overlap:
                    cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                    cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                    cxa = (x1_anc + x2_anc)/2.0
                    cya = (y1_anc + y2_anc)/2.0

                    tx = (cx - cxa) / (x2_anc - x1_anc)
                    ty = (cy - cya) / (y2_anc - y1_anc)
                    tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                    th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
                        
                if boundingBoxes[bbox_num]['class'] != 'bg':

                    # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                    if curr_iou > best_iou_for_bbox[bbox_num]:
                        best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                        best_iou_for_bbox[bbox_num] = curr_iou
                        best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
                        best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

                    # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                    if curr_iou > rpn_max_overlap:
                        bbox_type = 'pos'
                        num_anchors_for_bbox[bbox_num] += 1
                        # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                        if curr_iou > best_iou_for_loc:
                            best_iou_for_loc = curr_iou
                            best_regr = (tx, ty, tw, th)

                    # if the IOU is >min and <max, it is ambiguous and no included in the objective
                    if rpn_min_overlap < curr_iou < rpn_max_overlap:
                        # gray zone between neg and pos
                        if bbox_type != 'pos':
                            bbox_type = 'neutral'

            # turn on or off outputs depending on IOUs
            if bbox_type == 'neg':
                y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
            elif bbox_type == 'neutral':
                y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
            elif bbox_type == 'pos':
                y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                y_rpn_regr[jy, ix, start:start+4] = best_regr

def nonMaxSuppresionProcess(constants, shared_picked_boxes, shared_available_boxes, shared_last_index, lock, process_num):
    overlap_treshold = constants['overlap_treshold']
    area = constants['area']
    x1, y1, x2, y2 = constants['axes']
    max_boxes = constants['max_boxes']
    
    # Transform shared array to numpy
    np_picked_boxes = np.ctypeslib.as_array(shared_picked_boxes.get_obj())
    np_available_boxes = np.ctypeslib.as_array(shared_available_boxes.get_obj())

    while shared_last_index.value >= 0 and (np.count_nonzero(np_picked_boxes) < max_boxes):
        # Get the last working index
        with lock:
            keep_looking = True
            while keep_looking and shared_last_index.value >= 0:
                # If the index was not already selected or deleted, pick it
                if (np_available_boxes[shared_last_index.value]) and (not np_picked_boxes[shared_last_index.value]):
                    current_index = copy.copy(shared_last_index.value)
                    np_picked_boxes[current_index] = True
                    np_available_boxes[current_index] = False
                    keep_looking = False
                with shared_last_index.get_lock():
                    shared_last_index.value -= 1
    
        # Filter available indexes for IoU calculations (not deleted or picked yet)
        current_available_boxes = None
        with lock:
            current_available_boxes = np.copy(np_available_boxes).astype(np.bool)

        # Calculate overlaps
        xx1_int = np.maximum(x1[current_index], x1[current_available_boxes])
        yy1_int = np.maximum(y1[current_index], y1[current_available_boxes])
        xx2_int = np.minimum(x2[current_index], x2[current_available_boxes])
        yy2_int = np.minimum(y2[current_index], y2[current_available_boxes])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[current_index] + area[current_available_boxes] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6)

        # Find the indexes to delete
        overlaped_indexes = np.where(overlap > overlap_treshold)[0]
        available_indexes = np.where(current_available_boxes == True)[0]
        indexes_to_delete = available_indexes[overlaped_indexes]

        # Delete (flag as false) the indexes that surpass the treshold
        # Deselct boxes that are below the current (if any)
        with lock:
            np_available_boxes[indexes_to_delete] = False
            np_picked_boxes[indexes_to_delete] = False

def nonMaxSuppresionProcessIOU(constants, shared_picked_boxes, shared_available_boxes, shared_last_index,
                               shared_x_roi, shared_y_class_num, shared_y_class_regr_coords, shared_y_class_regr_label, shared_iou_items,
                               lock, process_num):
    # Load constants
    overlap_treshold = constants['overlap_treshold']
    area = constants['area']
    x1, y1, x2, y2 = constants['axes']
    max_boxes = constants['max_boxes']
    gta = constants['gta']
    bboxes = constants['bboxes']
    classifier_min_overlap = constants['classifier_min_overlap']
    classifier_max_overlap = constants['classifier_max_overlap']
    classifier_regr_std = constants['classifier_regr_std']   
    class_mapping =  constants['class_mapping']

    # Transform shared arrays to numpy
    np_picked_boxes = np.ctypeslib.as_array(shared_picked_boxes.get_obj())
    np_available_boxes = np.ctypeslib.as_array(shared_available_boxes.get_obj())
    np_iou_indexes = np.ctypeslib.as_array(shared_iou_items.get_obj())

    np_x_roi = np.ctypeslib.as_array(shared_x_roi.get_obj())
    np_x_roi = np_x_roi.reshape((len(x1), 4))

    np_y_class_num = np.ctypeslib.as_array(shared_y_class_num.get_obj())
    np_y_class_num = np_y_class_num.reshape((len(x1), len(class_mapping)))

    np_y_class_regr_coords = np.ctypeslib.as_array(shared_y_class_regr_coords.get_obj())
    np_y_class_regr_coords = np_y_class_regr_coords.reshape((len(x1), (4 * (len(class_mapping) - 1))))

    np_y_class_regr_label = np.ctypeslib.as_array(shared_y_class_regr_label.get_obj())
    np_y_class_regr_label = np_y_class_regr_label.reshape((len(x1), (4 * (len(class_mapping) - 1))))

    while shared_last_index.value >= 0 and (np.count_nonzero(np_picked_boxes) < max_boxes):
        # Get the last working index
        with lock:
            keep_looking = True
            while keep_looking and shared_last_index.value >= 0:
                # If the index was not already selected or deleted, pick it
                if (np_available_boxes[shared_last_index.value]) and (not np_picked_boxes[shared_last_index.value]):
                    current_index = copy.copy(shared_last_index.value)
                    np_picked_boxes[current_index] = True
                    np_available_boxes[current_index] = False
                    keep_looking = False
                with shared_last_index.get_lock():
                    shared_last_index.value -= 1
    
        # Filter available indexes for IoU calculations (not deleted or picked yet)
        current_available_boxes = None
        with lock:
            current_available_boxes = np.copy(np_available_boxes).astype(np.bool)

        # Calculate overlaps
        xx1_int = np.maximum(x1[current_index], x1[current_available_boxes])
        yy1_int = np.maximum(y1[current_index], y1[current_available_boxes])
        xx2_int = np.minimum(x2[current_index], x2[current_available_boxes])
        yy2_int = np.minimum(y2[current_index], y2[current_available_boxes])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[current_index] + area[current_available_boxes] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6)

        # Find the indexes to delete
        overlaped_indexes = np.where(overlap > overlap_treshold)[0]
        available_indexes = np.where(current_available_boxes == True)[0]
        indexes_to_delete = available_indexes[overlaped_indexes]

        # Delete (flag as false) the indexes that surpass the treshold
        # Deselct boxes that are below the current (if any)
        with lock:
            np_available_boxes[indexes_to_delete] = False
            np_picked_boxes[indexes_to_delete] = False

        #region Calculate IOU for the current box
        x1_current = copy.copy(int(round(x1[current_index])))
        y1_current = copy.copy(int(round(y1[current_index])))
        x2_current = copy.copy(int(round(x2[current_index])))
        y2_current = copy.copy(int(round(y2[current_index])))

        best_iou = 0.0
        best_bbox = -1
        for bbox_num in range(len(bboxes)):
            curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_current, y1_current, x2_current, y2_current])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        if best_iou < classifier_min_overlap:
            continue
        else:
            w = x2_current - x1_current
            h = y2_current - y1_current
            np_x_roi[current_index] = [x1_current, y1_current, w, h]
            np_iou_indexes[current_index] = True

            if classifier_min_overlap <= best_iou < classifier_max_overlap:
                # hard negative example
                cls_name = 'bg'
            elif classifier_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                cx = x1_current + w / 2.0
                cy = y1_current + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        # Classification per ROI (matching class index set to 1, all others to 0)
        class_num = class_mapping[cls_name]
        class_label = len(class_mapping) * [0]
        class_label[class_num] = 1 # 1 hot enconding
        np_y_class_num[current_index] = copy.deepcopy(class_label)
        coords = [0] * 4 * (len(class_mapping) - 1) # bg has no coords
        labels = [0] * 4 * (len(class_mapping) - 1)
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = classifier_regr_std
            coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
            labels[label_pos:4+label_pos] = [1, 1, 1, 1]
            np_y_class_regr_coords[current_index] = copy.deepcopy(coords)
            np_y_class_regr_label[current_index] = copy.deepcopy(labels)
        else:
            np_y_class_regr_coords[current_index] = copy.deepcopy(coords)
            np_y_class_regr_label[current_index] = copy.deepcopy(labels)
        #endregion
#endregion