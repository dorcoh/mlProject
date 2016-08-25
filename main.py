import math
import sys
import copy
import numpy as np
import itertools

# Load classes and functions from our modules
from DataStructures import TextData
from DataStructures import normaClassifier,svmClassifier
from Clique import sClassifier, superbClassifier,sClfAspect

"""  Main function """

if __name__ == '__main__':

    trainData = TextData('train-reviews.json',balance=False)
    testData = TextData('test-reviews.json',balance=False)

    xTrain, yTrain, docsTrain = trainData.x, trainData.y, trainData.docs
    xTest, yTest, docsTest = testData.x, testData.y, testData.docs

    # classifiers check
    svmClassifier(xTrain, yTrain, docsTrain, xTest, yTest, docsTest, testData.catCat)
    normaClassifier(xTrain, yTrain, docsTrain, xTest, yTest, docsTest)
    sClassifier(xTrain,yTrain,xTest,yTest)
    superbClassifier(xTrain,yTrain,xTest,yTest,[0,1,2,3])
    sClfAspect(xTrain,yTrain,xTest,yTest)