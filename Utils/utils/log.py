import logging
import os.path
import time


class Logging(object):
    """Class that logs information to console and files"""
    
    #region Constructor

    def __init__(self, logTypes, context):

        timestamp = time.strftime("%d-%b-%Y-%H-%M-%S", time.gmtime())

        self.LOGS = {
            'info':    { 'name': 'info-' + timestamp + '.log', 'logger': None },
            'metrics': { 'name': 'metrics-' + timestamp + '.log', 'logger': None }
        }

        for log in logTypes:
            self.createLogger(log, context)

    #endregion

    #region Methods

    def info(self, message):
        self.LOGS['info']['logger'].info(message)


    def metrics(self, message):
        self.LOGS['metrics']['logger'].info(message)


    def error(self, message, stackTrace=''):
        self.LOGS['info']['logger'].error('Exception: {} \n{}'.format(message, stackTrace))


    def createLogger(self, logKey, context):
        if logKey in self.LOGS:
            directory = context.getConfig('Logging', 'log_dir')

            formatter = logging.Formatter("<%(asctime)s> %(message)s")

            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(formatter)
            print(directory)
            fileHandler = logging.FileHandler(os.path.join(directory, self.LOGS[logKey]['name']), mode='a')
            fileHandler.setFormatter(formatter)

            self.LOGS[logKey]['logger'] = logging.getLogger(logKey)
            self.LOGS[logKey]['logger'].addHandler(consoleHandler)
            self.LOGS[logKey]['logger'].addHandler(fileHandler)
            self.LOGS[logKey]['logger'].setLevel(logging.INFO)
        else:
            raise Exception('No logger with that name is available')
    
    #endregion