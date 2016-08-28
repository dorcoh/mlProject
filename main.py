import math
import sys
import copy
import numpy as np
import itertools

# Load classes and functions from our modules
from DataStructures import TextData
from Classifiers import Directed,Undirected,NormaClassifier,SvmClassifier
from Analyze import plotStats

"""  Main function """

if __name__ == '__main__':
    
    # parse data
    trainData = TextData('train-reviews.json',balance=False)
    testData = TextData('test-reviews.json',balance=False)

    xTrain, yTrain, numDocsTrain = trainData.x, trainData.y, trainData.docs
    xTest, yTest, numDocsTest = testData.x, testData.y, testData.docs

    # assumption checking for norma classifier
    # plotStats(xTrain,yTrain,numDocsTrain,xTest,yTest,numDocsTest)

    # classify

    svm = SvmClassifier(xTrain,yTrain,xTest,yTest)
    svm.classify(catList=testData.catCat)

    norma = NormaClassifier(xTrain,yTrain,xTest,yTest)
    norma.classify(numDocsTrain,numDocsTest)
    
    dirClf = Directed(xTrain,yTrain,xTest,yTest)
    dirClf.classify(order=[0,1,2,3],verb=False)

    udirClf = Undirected(xTrain,yTrain,xTest,yTest)
    udirClf.classify(trainOn='all',epsilon=0.01)
    udirClf.classify(trainOn='aspect',epsilon=0.01)
