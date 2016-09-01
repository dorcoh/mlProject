import math
import sys
import copy
import numpy as np
import itertools

# Load classes and functions from our modules
from DataStructures import TextData
from Classifiers import Directed,Undirected,NormaClassifier,SvmClassifier
from Analyze import plotStats, outVar, plotMissHistogram, plotRatioHistogram

"""  Main function """

if __name__ == '__main__':
    
    # parse data from json files using TextData object
    # (also exclude first line of every paragraph (withFirst))
    trainData = TextData('train-reviews.json',withFirst=False)
    testData = TextData('test-reviews.json',withFirst=False)

    xTrain, yTrain, numDocsTrain = trainData.x, trainData.y, trainData.docs
    xTest, yTest, numDocsTest = testData.x, testData.y, testData.docs
    
    # ANALYZE DATA AND ASSUMPTION CHECKING
    # can uncomment to see plots
    """
    pos,neg = 0,0
    for i in range(len(xTrain)):
        if yTrain.score[i] == False:
            neg += 1
        else:
            pos += 1
    ratioDict = {
                    "Pos(Score>5)": float(pos)/len(xTrain), 
                    "Neg(Score<=5)": float(neg)/len(xTrain)
                }
    # pos/neg ratio
    plotRatioHistogram(ratioDict)

    # variance
    plotStats(xTrain,yTrain,numDocsTrain,xTest,yTest,numDocsTest)
    # 'real' variance
    print outVar(xTrain,yTrain)
    """
    # parameters for classifiers
    # exclude scores in this list
    exc = [5,6]
    # balance the classifiers (True/False ratio)
    bal = True

    print "Classifying: Excluding scores:{0}, balancing:{1}".format(
           exc,bal)

    # classify
    svm = SvmClassifier(xTrain,yTrain,xTest,yTest,exc,bal)
    missDict = svm.classify(catList=testData.catCat)

    # analyze misses of svm classifier
    #plotMissHistogram(missDict)

    # norma classifier
    #norma = NormaClassifier(xTrain,yTrain,xTest,yTest)
    #norma.classify(numDocsTrain,numDocsTest)
    
    # directed classifier
    dirClf = Directed(xTrain,yTrain,xTest,yTest,exc,bal)
    dirClf.classify([0,1,2,3],verb=False)

    # tune dependencies order in dirClf
    """ 
    for o in list(itertools.permutations([0,1,2,3])):
        print o
    dirClf.classify(order=list(o),verb=False)
    """

    udirClf = Undirected(xTrain,yTrain,xTest,yTest,exc,bal)
    udirClf.classify(trainOn='all',epsilon=0.0001)
    udirClf.classify(trainOn='aspect',epsilon=0.0001)
