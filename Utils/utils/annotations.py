import cv2
import numpy as np
import argparse
import json

def plotImageRegions(imagePath, annotationFilePath, newImageFilePath):
    img = cv2.imread(imagePath)

    with open(annotationFilePath) as annotationfile:
        count = 0
        for line in annotationfile:
            count += 1
            print(count)
            annotation = line.split(',')
            if len(annotation) >= 4:
                img = cv2.rectangle(img,(int(annotation[0]),int(annotation[1])),(int(annotation[2]),int(annotation[3])),(0,255,0),1)

    cv2.imwrite(newImageFilePath, img)


def convertImageRegions(imageName, annotationsTxtPath, annotationsJsonPath):
    with open(annotationsJsonPath, 'r') as jsonFile:
        jsonData = json.loads(jsonFile.read())

    # Find if image exists
    imageData = [x for x in jsonData['images'] if x['name'] == imageName]

    if len(imageData) == 0:
        print('Image not found, adding image to json')
        newImageData = {'name': imageName, 'objects': {}}
        jsonData['images'].append(newImageData)
        imageData = newImageData
    else:
        imageData = imageData[0]

    # Open annotations txt
    with open(annotationsTxtPath, 'r') as txtFile:
        count = 0
        for line in txtFile:
            count += 1
            annotation = line.split(',')
            if len(annotation) != 5:
                print('Error in annotation line {}, passing'.format(count))
                continue

            newObject = {'x1': int(annotation[0]), 'y1': int(annotation[1]), 'x2': int(annotation[2]), 'y2': int(annotation[3])}
            className = annotation[4].split('\n')[0]
            if className not in imageData['objects']:
                print('New object class, adding to json')
                imageData['objects'][className] = []

            imageData['objects'][className].append(newObject)

    print('writing file')
    with open(annotationsJsonPath, 'w') as jsonFile:
        json.dump(jsonData, jsonFile)

        
def calculateAvg(annotationFilePath):
    with open(annotationFilePath) as annotationfile:
        count = 0
        xAvg = 0
        yAvg = 0
        for line in annotationfile:
            annotation = line.split(',')
            
            x = int(annotation[2]) - int(annotation[0])
            y = int(annotation[3]) - int(annotation[1])
            if x > y:
                xTemp = x
                x = y
                y = xTemp
                
            xAvg += x
            yAvg += y
            
            count += 1
        
        xAvg = xAvg / count
        yAvg = yAvg / count
        
        print('X: ' + str(xAvg))
        print('Y: ' + str(yAvg))


def resizeImage(imagePath, outputImagePath, resizeFactorX, resizeFactorY):
    img = cv2.imread(imagePath)
    resizedimg = cv2.resize(img, (0,0), fx=resizeFactorX, fy=resizeFactorY)
    cv2.imwrite(outputImagePath, resizedimg)


def cutAnnotationsInBounds(annotationFilePath, newAnnotationFilePath, x1_bound, y1_bound, x2_bound, y2_bound):
    with open(annotationFilePath) as annotationfile:
        with open(newAnnotationFilePath, 'w') as newAnnotationFile:
            count = 0
            for line in annotationfile:
                count += 1
                print(count)
                annotation = line.split(',')
                if len(annotation) >= 4:
                    if (int(annotation[0]) > x1_bound) and (int(annotation[0]) < x2_bound) and (int(annotation[1]) > y1_bound) and (int(annotation[1]) < y2_bound) and (int(annotation[2]) > x1_bound) and (int(annotation[2]) < x2_bound) and (int(annotation[3]) > y1_bound) and (int(annotation[3]) < y2_bound):
                        x1 = int(annotation[0]) - x1_bound
                        y1 = int(annotation[1]) - y1_bound
                        x2 = int(annotation[2]) - x1_bound
                        y2 = int(annotation[3]) - y1_bound
                        if len(annotation) == 4:
                            newAnnotationFile.write("{},{},{},{}".format(x1,y1,x2,y2))
                        elif len(annotation) == 5:
                            newAnnotationFile.write("{},{},{},{},{}".format(x1,y1,x2,y2,annotation[4]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Annotation plotter\n")
    parser.add_argument('-c', type=str, choices=['plot', 'convert', 'average', 'resize', 'cut'], help='Operation to perform.\n', required=True)
    parser.add_argument('-i', type=str, help='Image file path or image name.\n')
    parser.add_argument('-f', type=str, help='Path to txt annotations\n')
    parser.add_argument('-fx', type=str, help='Path to txt annotations\n')
    parser.add_argument('-fy', type=str, help='Path to txt annotations\n')
    parser.add_argument('-o', type=str, help='Output file for annotated image\n')
    parser.add_argument('-b', type=str, help='X1,Y1,X2,Y2 bounds of the image to get the annotations\n')

    args = vars(parser.parse_args())
    
    option = args.get('c')

    try:
        if option == 'plot':
            imagePath = args.get('i')
            annotationsPath = args.get('f')
            newImageFilePath = args.get('o')
            plotImageRegions(imagePath, annotationsPath, newImageFilePath)
        elif option == 'convert':
            imageName = args.get('i')
            annotationsJsonPath = args.get('o')
            annotationsTxtPath = args.get('f')
            convertImageRegions(imageName, annotationsTxtPath, annotationsJsonPath)
        elif option == 'average':
            annotationsTxtPath = args.get('f')
            calculateAvg(annotationsTxtPath)
        elif option == 'resize':
            imageName = args.get('i')
            resizeFactorX = float(args.get('fx'))
            resizeFactorY = float(args.get('fy'))
            newImageFilePath = args.get('o')
            resizeImage(imageName, newImageFilePath, resizeFactorX, resizeFactorY)
        elif option == 'cut':
            annotationFile = args.get('f')
            newAnnotationFile = args.get('o')
            bounds = args.get('b')
            bounds = bounds.split(',')
            for i in range(len(bounds)):
                bounds[i] = int(bounds[i])
            cutAnnotationsInBounds(annotationFile, newAnnotationFile, bounds[0], bounds[1], bounds[2], bounds[3])

    except Exception as e:
        print('Error when running command {}:{} \n{}'.format(option, e, traceback.format_exc()))