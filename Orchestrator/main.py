import sys
sys.path.append('../Data')
sys.path.append('../DetectionEngine')
sys.path.append('../Utils')

import argparse
import configparser
import os
import traceback

import entities.context
import utils.log


def setContextParameters(context, rootDir):
    """Override configuration parameters if given"""

    if args.get('num-rois'):
        context.configuration.set('NeuralNetwork', 'num_rois', args.get('num-rois'))

    if args.get('num-epochs'): 
        context.configuration.set('NeuralNetwork', 'num_epochs', args.get('num-epochs'))

    if args.get('i'):
        if not os.path.isabs('w'):
            abspath = os.path.join(rootDir, args.get('w'))
        else:
            abspath = args.get('w')
            context.configuration.set('NeuralNetwork', 'weigths_path', abspath)

    if args.get('o'):
        if not os.path.isabs('o'):
            abspath = os.path.join(rootDir, args.get('o'))
        else:
            abspath = args.get('o')
            context.configuration.set('NeuralNetwork', 'output_path', abspath)


def main(args, rootDir):
    # Import is done here to avoid library re-initialization with mutiprocessing
    import orchestrator
    try:
        # Load context / configuration
        configPath = args.get('config')
        dataPath = args.get('p')
        if not os.path.isabs(configPath):
            configPath = os.path.join(rootDir, configPath)
        print(configPath)
        context = entities.context.Context(configPath)
        context.dataPath = dataPath
        print(configPath)
        setContextParameters(context, rootDir)
        
        command = args.get('c')

        # Do job
        if command == 'train':
            logger = utils.log.Logging(['info'], context)
            orchest = orchestrator.Orchestrator(logger)
            orchest.train(context)

        elif command == 'detect':
            logger = utils.log.Logging(['info'], context)
            orchest = orchestrator.Orchestrator(logger)
            orchest.detect(context)
        elif command == 'measure':
            print('3')


    except Exception as e:
        try:
            logger.error(e, traceback.format_exc())
        except Exception as e1:
            print('Exception1:', e,  traceback.format_exc())
            print('Exception2: ', e1, traceback.format_exc())



if __name__ == '__main__':
    sys.setrecursionlimit(40000)
    rootDir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser("Detection Engine\n")
    parser.add_argument('-c', type=str, choices=['train', 'detect', 'measure'], help='Operation to perform.  i.e: -c train\n', required=True)
    
    parser.add_argument('-p', type=str, help='Path to train annotations or test image\n', required=True)
    parser.add_argument('-config', type=str, help='Path to test/training configuration file\n', default=os.path.join(rootDir,'configuration.cfg'))
    parser.add_argument('-w', type=str, help='Path to pre-trained model weights\n', default=os.path.join(rootDir, 'weights.h5'))
    parser.add_argument('-o', type=str, help='Output file for trained model\n', default=os.path.join(rootDir, 'model.hdf5'))

    parser.add_argument('-num-rois', type=int, help='Number of simultaneous ROIs to process in classifier\n')
    parser.add_argument('-num-epochs', type=int, help='Number of full train iterations to perform\n')

    args = vars(parser.parse_args())


    main(args, rootDir)