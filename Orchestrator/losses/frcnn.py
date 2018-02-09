from keras import backend as K
from keras.objectives import categorical_crossentropy
import tensorflow as tf
import numpy as np

import losses.base
import json

class FasterRCNNLoss(losses.base.Loss):
    """Implementation of Faster RCNN algorithm loss calculations
        Code based from: https://github.com/yhenon/keras-frcnn
    """

    def __init__(self, epoch_length):
        self.lambda_rpn_regr = 1.0
        self.lambda_rpn_class = 1.0
        self.lambda_cls_regr = 1.0
        self.lambda_cls_class = 1.0
        self.epsilon = 1e-4

        # Arrays to store training loss numbers
        self.loss_rpn_cls = np.zeros(epoch_length)
        self.loss_rpn_regr = np.zeros(epoch_length)
        self.loss_class_cls = np.zeros(epoch_length)
        self.loss_class_regr = np.zeros(epoch_length)
        self.class_accuracy = np.zeros(epoch_length)


    def rpnRegressionLoss(self, context):
        num_anchors = len(json.loads(context.getConfig('NeuralNetwork', 'anchor_scales'))) * len(json.loads(context.getConfig('NeuralNetwork', 'anchor_ratios')))

        def rpn_loss_regr_fixed_num(y_true, y_pred):
            x = y_true[:, :, :, 4 * num_anchors:] - y_pred
            x_abs = K.abs(x)
            x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

            return self.lambda_rpn_regr * K.sum(
                y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(self.epsilon + y_true[:, :, :, :4 * num_anchors])

        return rpn_loss_regr_fixed_num


    def rpnClassLoss(self, context):
        num_anchors = len(json.loads(context.getConfig('NeuralNetwork', 'anchor_scales'))) * len(json.loads(context.getConfig('NeuralNetwork', 'anchor_ratios')))

        def rpn_loss_cls_fixed_num(y_true, y_pred):
            return self.lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(self.epsilon + y_true[:, :, :, :num_anchors])

        return rpn_loss_cls_fixed_num


    def classifierRegressionLoss(self, context):
        # TODO: Get number of classes in a better way
        num_classes = len(context.classCount) - 1

        def class_loss_regr_fixed_num(y_true, y_pred):
            x = y_true[:, :, 4*num_classes:] - y_pred
            x_abs = K.abs(x)
            x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
            return self.lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(self.epsilon + y_true[:, :, :4*num_classes])
        return class_loss_regr_fixed_num


    def classifierClassLoss(self, context):
        def class_loss_cls(y_true, y_pred):
            return self.lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))

        return class_loss_cls