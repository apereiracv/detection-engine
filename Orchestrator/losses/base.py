class Loss(object):
    """Base class for loss calculations"""
    
    def rpnClassLoss(self, context):
        return NotImplementedError()

    def rpnRegressionLoss(self, context):
        return NotImplementedError()

    def classifierClassLoss(self, context):
        return NotImplementedError()

    def classifierRegressionLoss(self, context):
        return NotImplementedError()
