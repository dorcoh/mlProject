import math
import sys
import copy
import numpy as np
import itertools
import time

# Load classes and functions from our modules
from DataStructures import TextData, Feature
from Classifiers import Directed,Undirected,NormaClassifier,SvmClassifier
from Analyze import plotStats, outVar, plotHistogram, posNegRatio
from Tune import tuneUndirected, tuneDirected

"""  Main function """

if __name__ == '__main__':
    
    # -------------  PARSING DATA --------------------
    # ------------------------------------------------

    # parse data from json files using TextData object
    # (also exclude first line of every paragraph (withFirst))
    trainData = TextData('train-reviews.json',withFirst=False)
    testData = TextData('test-reviews.json',withFirst=False)

    xTrain, yTrain, numDocsTrain = trainData.x, trainData.y, trainData.docs
    xTest, yTest, numDocsTest = testData.x, testData.y, testData.docs
    
    # -------------  ANALYZE DATA --------------------
    # ------------------------------------------------
    # uncomment to see plots
    
    # pos/neg ratio
    #posNegRatio(xTrain,yTrain)
    # variance
    #plotStats(xTrain,yTrain,numDocsTrain,xTest,yTest,numDocsTest)
    # 'real' variance
    #print outVar(xTrain,yTrain)

    # -------------  CLASSIFICATION ------------------
    # ------------------------------------------------

    # parameters for classifiers
    
    # exclude scores in this list
    exc = [5,6]
    # balance the classifiers (True/False ratio)
    bal = True

    print "Classifying: Excluding scores:{0}, balancing:{1}".format(
           exc,bal)

    # -------------  SVM BASELINE ------------------

    svm = SvmClassifier(xTrain,yTrain,xTest,yTest,exc,bal)
    missDict,occDict = svm.classify(catList=testData.catCat)

    # plot misses/occur histograms
    #plotHistogram(occDict)
    #plotHistogram(missDict)
    
    # -------------  DIRECTED ----------------------

    # TUNE dependencies order in directed classifier
    #o = tuneDirected(xTrain,yTrain,xTest,yTest,exc,bal)

    # Directed CLF - 
    # using tuned parameters
    tunedOrder = [2,0,1,3]
    dirClf = Directed(xTrain,yTrain,xTest,yTest,exc,bal)
    dirClf.classify(tunedOrder,verb=False)

    # -------------  UNDIRECTED --------------------

    # TUNE undirected classifier
    # by assigning weights to the probabilities
    # initParams = [[0.3,0.3,0.4],[0.3,0.3,0.4],[0.3,0.3,0.4],[0.3,0.3,0.4]]
    # params = tuneUndirected(xTrain,yTrain,initParams,exc,bal,foldTo=4)
    
    # parameters (weights) for classifier

    # default (no weights)
    defParams = [[1,1,1],[1,1,1],[1,1,1],[1,1,1]]
    # TUNED parameters (with Excluding=True,Balance=True)
    if exc and bal:
        paramsB = [[0.0, 0.46666666666666656, 0.5333333333333333],
                  [0.0, 0.0, 1.0], [0.3, 0.3, 0.4],
                  [0.3, 0.3, 0.4]]
    else:
        paramsB = defParams

    # Undirected CLF - A 
    udirClf = Undirected(xTrain,yTrain,xTest,yTest,defParams,exc,bal) 
    udirClf.classify(trainOn='all',epsilon=0.0001)
    # Undirected CLF - B (trained by specific aspect)
    # Using tuned parameters
    udirClf = Undirected(xTrain,yTrain,xTest,yTest,paramsB,exc,bal) 
    udirClf.classify(trainOn='aspect',epsilon=0.0001)

    # -------------  NORMA ------------------------

    # Norma CLF (EXCLUDED FROM OUR RESULTS)
    #norma = NormaClassifier(xTrain,yTrain,xTest,yTest)
    #norma.classify(numDocsTrain,numDocsTest)