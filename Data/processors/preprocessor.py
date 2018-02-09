import cv2
import numpy as np
import multiprocessing
import json
import random
import copy
import traceback
import os
import pickle
import ast

import entities.image

class ImagePreprocessor(object):
    
    def __init__(self, logger, neuralNetwork):
        self.logger = logger
        self.getFeatureMapSize = neuralNetwork.getFeatureMapSize

    #region Overlap calculations
    def intersection(self, ai, bi):
        x = max(ai[0], bi[0])
        y = max(ai[1], bi[1])
        w = min(ai[2], bi[2]) - x
        h = min(ai[3], bi[3]) - y
        if w < 0 or h < 0:
            return 0
        return w*h

    def iou(self, a, b):
        # a and b should be (x1,y1,x2,y2)
        if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
            return 0.0

        area_i = self.intersection(a, b)
        area_u = self.union(a, b, area_i)

        return float(area_i) / float(area_u) #+ 1e-6

    def union(self, au, bu, area_intersection):
        area_a = (au[2] - au[0]) * (au[3] - au[1])
        area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
        area_union = area_a + area_b - area_intersection
        return area_union

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
            x1 = int(round(x1))
            y1 = int(round(y1))
            w1 = int(round(w1))
            h1 = int(round(h1))
        
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

            x1 = np.round(x1)
            y1 = np.round(y1)
            w1 = np.round(w1)
            h1 = np.round(h1)
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


    def getImageDataIterator(self, imageList, context):
        while True:
            # TODO: Randomize
            for image in imageList:
                self.logger.info('Loading image: {}'.format(image.filePath))
                image.imageData = cv2.imread(image.filePath)
                (image.height, image.width, _) = image.imageData.shape

                resizeFactor = float(context.getConfig('Preprocessing', 'resize_factor'))
                # Resize to scaling factor and normalize pixels range
                image.resizedImageData, image.resizedWidth, image.resizedHeight = self.resizeImage(image, resizeFactor) 
                image.resizedImageData = self.normalizeImageData(image.resizedImageData, context)

                # Calculate RPN regions for training objective
                # TODO: Save calculated regions into file
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
        filename = image.id + '_s' + stride + '_r' + resize + '.pickle'
        
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

        downscale = float(context.getConfig('NeuralNetwork', 'rpn_stride'))
        anchor_sizes = json.loads(context.getConfig('NeuralNetwork', 'anchor_scales'))
        anchor_ratios = json.loads(context.getConfig('NeuralNetwork', 'anchor_ratios'))

        num_anchors = len(anchor_sizes) * len(anchor_ratios)	

        # calculate the output feature map size based on the network architecture
        featureMapWidth, featureMapHeight = self.getFeatureMapSize(image.resizedWidth, image.resizedHeight)
    
        num_bboxes = len(image.boundingBoxes)

        # Initialise bounding box control arrays
        num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
        best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
        best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
        best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
        best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

        # initialise empty output objectives
        y_rpn_overlap = np.zeros((featureMapHeight, featureMapWidth, num_anchors))
        y_is_box_valid = np.zeros((featureMapHeight, featureMapWidth, num_anchors))
        y_rpn_regr = np.zeros((featureMapHeight, featureMapWidth, num_anchors * 4))

        # get the GT box coordinates, and resize to account for image resizing
        gta = np.zeros((num_bboxes, 4))
        for bbox_num, bbox in enumerate(image.boundingBoxes):
            # get the GT box coordinates, and resize to account for image resizing
            gta[bbox_num, 0] = bbox['x1'] * (image.resizedWidth / float(image.width))
            gta[bbox_num, 1] = bbox['x2'] * (image.resizedWidth / float(image.width))
            gta[bbox_num, 2] = bbox['y1'] * (image.resizedHeight / float(image.height))
            gta[bbox_num, 3] = bbox['y2'] * (image.resizedHeight / float(image.height))
    
        n_anchratios = len(anchor_ratios)
        # Create constant values dictionary
        constants = {
            'downscale' : int(context.getConfig('NeuralNetwork','rpn_stride')),
            'rpn_min_overlap': float(context.getConfig('NeuralNetwork','rpn_min_overlap')),
            'rpn_max_overlap': float(context.getConfig('NeuralNetwork','rpn_max_overlap')),
            'anchor_sizes' : anchor_sizes,
            'anchor_ratios': anchor_ratios,
            'output_width' : featureMapWidth,
            'output_height': featureMapHeight,
            'resized_width' : image.resizedWidth,
            'resized_height': image.resizedHeight,
            'n_anchratios': n_anchratios,
            'gta': gta,
            'boundingBoxes': image.boundingBoxes,
            'num_bboxes': num_bboxes,
            'num_anchors': num_anchors
        }

        # Create shared arrays for multiprocessing
        ctype_y_rpn_overlap  = y_rpn_overlap.flatten().tolist()
        shared_y_rpn_overlap = multiprocessing.Array('d', ctype_y_rpn_overlap)

        ctype_y_is_box_valid = y_is_box_valid.flatten().tolist()
        shared_y_is_box_valid = multiprocessing.Array('d', ctype_y_is_box_valid)

        ctype_y_rpn_regr = y_rpn_regr.flatten().tolist()
        shared_y_rpn_regr = multiprocessing.Array('d', ctype_y_rpn_regr)

        ctype_num_anchors_for_bbox = np.ctypeslib.as_ctypes(num_anchors_for_bbox.flatten())
        shared_num_anchors_for_bbox = multiprocessing.Array(ctype_num_anchors_for_bbox._type_, ctype_num_anchors_for_bbox)

        ctype_best_anchor_for_bbox = np.ctypeslib.as_ctypes(best_anchor_for_bbox.flatten())
        shared_best_anchor_for_bbox = multiprocessing.Array(ctype_best_anchor_for_bbox._type_, ctype_best_anchor_for_bbox)

        ctype_best_iou_for_bbox = np.ctypeslib.as_ctypes(best_iou_for_bbox.flatten())
        shared_best_iou_for_bbox = multiprocessing.Array(ctype_best_iou_for_bbox._type_, ctype_best_iou_for_bbox)

        ctype_best_x_for_bbox = np.ctypeslib.as_ctypes(best_x_for_bbox.flatten())
        shared_best_x_for_bbox = multiprocessing.Array(ctype_best_x_for_bbox._type_, ctype_best_x_for_bbox)

        ctype_best_dx_for_bbox = np.ctypeslib.as_ctypes(best_dx_for_bbox.flatten())
        shared_best_dx_for_bbox = multiprocessing.Array(ctype_best_dx_for_bbox._type_, ctype_best_dx_for_bbox)

        # Calculate regions for each anchor on separate processes
        anchor_processes = []
        for anchor_size_idx in range(len(anchor_sizes)):
            for anchor_ratio_idx in range(n_anchratios):
                # Create a new thread object for each anchor, with it's shared arguments
                single_anchor_process = multiprocessing.context.Process(target = calculateRPNProcess, args = (constants, anchor_size_idx, anchor_ratio_idx,
                                    shared_y_rpn_overlap, shared_y_is_box_valid, shared_y_rpn_regr, shared_num_anchors_for_bbox, 
                                    shared_best_anchor_for_bbox, shared_best_iou_for_bbox, shared_best_x_for_bbox, shared_best_dx_for_bbox))
            
                anchor_processes.append(single_anchor_process)

        # Start all threads
        for process in anchor_processes:
            process.start()

        # Wait for all threads to finish
        for process in anchor_processes:
            process.join()

        # Recapture modified arrays into their original types / shapes
        y_rpn_overlap = np.copy(np.ctypeslib.as_array(shared_y_rpn_overlap.get_obj()))
        y_rpn_overlap = y_rpn_overlap.reshape((featureMapHeight, featureMapWidth, num_anchors))

        y_is_box_valid = np.copy(np.ctypeslib.as_array(shared_y_is_box_valid.get_obj()))
        y_is_box_valid = y_is_box_valid.reshape((featureMapHeight, featureMapWidth, num_anchors))

        y_rpn_regr = np.copy(np.ctypeslib.as_array(shared_y_rpn_regr.get_obj()))
        y_rpn_regr = y_rpn_regr.reshape((featureMapHeight, featureMapWidth, num_anchors * 4))

        num_anchors_for_bbox = np.copy(np.ctypeslib.as_array(shared_num_anchors_for_bbox.get_obj()))

        best_anchor_for_bbox = np.copy(np.ctypeslib.as_array(shared_best_anchor_for_bbox.get_obj()))
        best_anchor_for_bbox = best_anchor_for_bbox.reshape((num_bboxes, 4))

        best_iou_for_bbox = np.copy(np.ctypeslib.as_array(shared_best_iou_for_bbox.get_obj()))

        best_x_for_bbox = np.copy(np.ctypeslib.as_array(shared_best_x_for_bbox.get_obj()))
        best_x_for_bbox = best_x_for_bbox.reshape((num_bboxes, 4))

        best_dx_for_bbox = np.copy(np.ctypeslib.as_array(shared_best_dx_for_bbox.get_obj()))
        best_dx_for_bbox = best_dx_for_bbox.reshape((num_bboxes, 4))

        # we ensure that every bbox has at least one positive RPN region
        for idx in range(num_anchors_for_bbox.shape[0]):
            if num_anchors_for_bbox[idx] == 0:
                # no box with an IOU greater than zero ...
                if best_anchor_for_bbox[idx, 0] == -1:
                    continue
                y_is_box_valid[
                    best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                    best_anchor_for_bbox[idx,3]] = 1
                y_rpn_overlap[
                    best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                    best_anchor_for_bbox[idx,3]] = 1
                start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
                y_rpn_regr[
                    best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

        y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
        y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

        y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
        y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

        y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
        y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

        pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
        neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))
    
        # TODO: Un-limit regions
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


    def rpnToROI(self, image, context, calculate_iou = False):
        std_scaling = float(context.getConfig('Preprocessing', 'std_scaling'))
        anchor_sizes = json.loads(context.getConfig('NeuralNetwork', 'anchor_scales'))
        anchor_ratios = json.loads(context.getConfig('NeuralNetwork', 'anchor_ratios'))
        rpn_stride = int(context.getConfig('NeuralNetwork', 'rpn_stride'))

        rpn_layer = image.predictedRegions[0]
        regr_layer = image.predictedRegions[1] / std_scaling
        
        assert image.predictedRegions[0].shape[0] == 1

        (rows, cols) = image.predictedRegions[0].shape[1:3]

        curr_layer = 0
        A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))
        
        for anchor_size in anchor_sizes:
            for anchor_ratio in anchor_ratios:

                anchor_x = (anchor_size * anchor_ratio[0])/rpn_stride
                anchor_y = (anchor_size * anchor_ratio[1])/rpn_stride

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

        if calculate_iou:
            result = self.nonMaxSuppressionWithIOU(image, all_boxes, all_probs, context)
        else:
            result = self.nonMaxSuppression(image, all_boxes, all_probs, context)

        return result


    def nonMaxSuppression(self, boxes, probs, context):
        # code example from: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        overlap_thresh = float(context.getConfig('NeuralNetwork', 'suppresion_overlap'))
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


    def calculateIOU(self, R, image, class_mapping, context):
        rpn_stride = int(context.getConfig('NeuralNetwork', 'rpn_stride'))
        classifier_min_overlap = float(context.getConfig('NeuralNetwork', 'classifier_min_overlap'))
        classifier_max_overlap = float(context.getConfig('NeuralNetwork', 'classifier_max_overlap'))
        classifier_regr_std = json.loads(context.getConfig('Preprocessing', 'classifier_regr_std'))

        bboxes = image.boundingBoxes
        (width, height) = (image.width, image.height)
        (resized_width, resized_height) = (image.resizedWidth, image.resizedHeight)

        gta = np.zeros((len(bboxes), 4))

        for bbox_num, bbox in enumerate(bboxes):
            # get the GT box coordinates, and resize to account for image resizing
            gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/rpn_stride))
            gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/rpn_stride))
            gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/rpn_stride))
            gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/rpn_stride))

        x_roi = []
        y_class_num = []
        y_class_regr_coords = []
        y_class_regr_label = []

        for ix in range(R.shape[0]):
            (x1, y1, x2, y2) = R[ix, :]
            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))

            best_iou = 0.0
            best_bbox = -1
            for bbox_num in range(len(bboxes)):
                curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])
                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_bbox = bbox_num

            if best_iou < classifier_min_overlap:
                continue
            else:
                w = x2 - x1
                h = y2 - y1
                x_roi.append([x1, y1, w, h])

                if classifier_min_overlap <= best_iou < classifier_max_overlap:
                    # hard negative example
                    cls_name = 'bg'
                elif classifier_max_overlap <= best_iou:
                    cls_name = bboxes[best_bbox]['class']
                    cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                    cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                    cx = x1 + w / 2.0
                    cy = y1 + h / 2.0

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
            y_class_num.append(copy.deepcopy(class_label))
            coords = [0] * 4 * (len(class_mapping) - 1) # bg has no coords
            labels = [0] * 4 * (len(class_mapping) - 1)
            if cls_name != 'bg':
                label_pos = 4 * class_num
                sx, sy, sw, sh = classifier_regr_std
                coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
                labels[label_pos:4+label_pos] = [1, 1, 1, 1]
                y_class_regr_coords.append(copy.deepcopy(coords))
                y_class_regr_label.append(copy.deepcopy(labels))
            else:
                y_class_regr_coords.append(copy.deepcopy(coords))
                y_class_regr_label.append(copy.deepcopy(labels))

        if len(x_roi) == 0:
            return None, None, None, None
        X = np.array(x_roi)
        Y1 = np.array(y_class_num)
        Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)],axis=1)

        return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0)


    def nonMaxSuppressionWithIOU(self, image, predicted_boxes, predicted_probs, context):
        # code example from: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        # if there are no boxes, return an empty list
        if len(predicted_boxes) == 0:
            return []

        # Get context config
        suppresion_overlap = float(context.getConfig('NeuralNetwork', 'suppresion_overlap'))
        max_boxes = int(context.getConfig('NeuralNetwork', 'max_boxes'))
        class_mapping = ast.literal_eval(context.getConfig('Training', 'class_mapping'))
        rpn_stride = int(context.getConfig('NeuralNetwork', 'rpn_stride'))
        classifier_min_overlap = float(context.getConfig('NeuralNetwork', 'classifier_min_overlap'))
        classifier_max_overlap = float(context.getConfig('NeuralNetwork', 'classifier_max_overlap'))
        classifier_regr_std = json.loads(context.getConfig('Preprocessing', 'classifier_regr_std'))

        #region NonMaxSuppresion stuff
        # Sort bounding boxes by probabilities
        idxs = np.argsort(predicted_probs)
        predicted_boxes = predicted_boxes[idxs]
        predicted_probs = predicted_probs[idxs]
        self.logger.info('Total proposed regions: {}'.format(len(predicted_boxes)))

        # grab the coordinates of the bounding boxes
        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if predicted_boxes.dtype.kind == "i":
            predicted_boxes = predicted_boxes.astype("float")

        x1 = predicted_boxes[:, 0]
        y1 = predicted_boxes[:, 1]
        x2 = predicted_boxes[:, 2]
        y2 = predicted_boxes[:, 3]
        np.testing.assert_array_less(x1, x2)
        np.testing.assert_array_less(y1, y2)

        # calculate the areas
        area = (x2 - x1) * (y2 - y1)
        #endregion

        bboxes = image.boundingBoxes
        (width, height) = (image.width, image.height)
        (resized_width, resized_height) = (image.resizedWidth, image.resizedHeight)
        gta = np.zeros((len(bboxes), 4))

        # Calculate resized ground truth boxes
        for bbox_num, bbox in enumerate(bboxes):
            # get the GT box coordinates, and resize to account for image resizing
            gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/rpn_stride))
            gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/rpn_stride))
            gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/rpn_stride))
            gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/rpn_stride))


        # Create constants dict for processes
        constants = {
            'overlap_treshold' : suppresion_overlap,
            'area' : area,
            'axes' : (x1, y1, x2, y2),
            'max_boxes': max_boxes,
            'gta': gta,
            'bboxes': bboxes,
            'classifier_min_overlap': classifier_min_overlap,
            'classifier_max_overlap': classifier_max_overlap,
            'classifier_regr_std': classifier_regr_std,
            'class_mapping': class_mapping
        }

        # Create shared variables between processes
        # NMS arrays
        shared_last_index = multiprocessing.Value('i', (len(idxs) - 1))

        deleted_items = np.ones(len(idxs), dtype=np.bool).tolist()
        picked_items = np.zeros(len(idxs), dtype=np.bool).tolist()
        iou_items = np.zeros(len(idxs), dtype=np.bool).tolist()

        shared_deleted_items = multiprocessing.Array('b', deleted_items)
        shared_picked_items = multiprocessing.Array('b', picked_items)
        shared_iou_items = multiprocessing.Array('b', iou_items)

         # IOU arrays
        x_roi = np.zeros((len(predicted_boxes), 4))
        y_class_num = np.zeros((len(predicted_boxes), len(class_mapping)))
        y_class_regr_coords = np.zeros((len(predicted_boxes), (4 * (len(class_mapping) - 1))))
        y_class_regr_label = np.zeros((len(predicted_boxes), (4 * (len(class_mapping) - 1))))

        ctype_x_roi = x_roi.flatten().tolist()
        shared_x_roi = multiprocessing.Array('d', ctype_x_roi)

        ctype_y_class_num = y_class_num.flatten().tolist()
        shared_y_class_num = multiprocessing.Array('d', ctype_y_class_num)

        ctype_y_class_regr_coords = y_class_regr_coords.flatten().tolist()
        shared_y_class_regr_coords = multiprocessing.Array('d', ctype_y_class_regr_coords)

        ctype_y_class_regr_label = y_class_regr_label.flatten().tolist()
        shared_y_class_regr_label = multiprocessing.Array('d', ctype_y_class_regr_label)

        lock = multiprocessing.Lock()
        num_processes = multiprocessing.cpu_count()
        processes_array = []

        for process_num in range(num_processes):
            nms_process = multiprocessing.context.Process(target = nonMaxSuppresionProcessIOU, 
                                                          args = (constants, shared_picked_items, shared_deleted_items, shared_last_index,
                                                                  shared_x_roi, shared_y_class_num, shared_y_class_regr_coords, shared_y_class_regr_label, shared_iou_items,
                                                                  lock, process_num))
            processes_array.append(nms_process)

        # Start all processes
        for process in processes_array:
            process.start()

        # Wait for all processes to finish
        for process in processes_array:
            process.join()

        # Return only the picked boxes from the indexes
        final_picked_indexes = np.copy(np.ctypeslib.as_array(shared_picked_items.get_obj())).astype(np.bool)
        final_iou_indexes = np.copy(np.ctypeslib.as_array(shared_iou_items.get_obj())).astype(np.bool)

        resultBoxes = predicted_boxes[final_picked_indexes].astype("int")
        resultProbs = predicted_probs[final_picked_indexes]

        # Recapture modified arrays of IOU calculations
        x_roi = np.copy(np.ctypeslib.as_array(shared_x_roi.get_obj()))
        x_roi = x_roi.reshape((len(predicted_boxes), 4))

        y_class_num = np.copy(np.ctypeslib.as_array(shared_y_class_num.get_obj()))
        y_class_num = y_class_num.reshape((len(predicted_boxes), len(class_mapping)))

        y_class_regr_coords = np.copy(np.ctypeslib.as_array(shared_y_class_regr_coords.get_obj()))
        y_class_regr_coords = y_class_regr_coords.reshape((len(predicted_boxes), (4 * (len(class_mapping) - 1))))

        y_class_regr_label = np.copy(np.ctypeslib.as_array(shared_y_class_regr_label.get_obj()))
        y_class_regr_label = y_class_regr_label.reshape((len(predicted_boxes), (4 * (len(class_mapping) - 1))))

        result_y_class_regr_coords = y_class_regr_coords[final_iou_indexes]
        result_y_class_regr_label = y_class_regr_label[final_iou_indexes]
        
        X = x_roi[final_iou_indexes]
        Y1 = y_class_num[final_iou_indexes]
        Y2 = np.concatenate([result_y_class_regr_label, result_y_class_regr_coords], axis=1)

        return resultBoxes, resultProbs, np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0)
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