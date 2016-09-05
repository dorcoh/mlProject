import math
import sys
import copy
import numpy as np
import itertools
import time

# Load classes and functions from our modules
from DataStructures import TextData, Feature
from Classifiers import Directed,Undirected,NormaClassifier,SvmClassifier
from Analyze import plotStats, outVar, plotMissHistogram, plotRatioHistogram

def normalizeWeight(tup):
    # normalize the weights tuple
    s = 0
    for i in range(len(tup)):
        s += tup[i]

    newTup = [x/float(s) for x in tup]
    return newTup

def splitData(xTrain,yTrain,k,p):
    # split the data into k parts
    length = len(xTrain)/4
    portion = length / k
    newXTrain, newXTest = [], []
    newYTrain, newYTest = Feature(), Feature()
    for i in range(len(xTrain)):
        # check for the correct portion
        if i/4 >= p*portion and i/4 < (p+1)*portion:
            # test
            newXTest.append(xTrain[i])
            newYTest.doc.append(yTrain.doc[i])
            newYTest.cat.append(yTrain.cat[i])
            newYTest.score.append(yTrain.score[i])
            newYTest.trueScore.append(yTrain.trueScore[i])
        else:
            # train
            newXTrain.append(xTrain[i])
            newYTrain.doc.append(yTrain.doc[i])
            newYTrain.cat.append(yTrain.cat[i])
            newYTrain.score.append(yTrain.score[i])
            newYTrain.trueScore.append(yTrain.trueScore[i])

    return newXTrain, newYTrain, newXTest, newYTest

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
    #svm = SvmClassifier(xTrain,yTrain,xTest,yTest,exc,bal)
    #missDict = svm.classify(catList=testData.catCat)

    # analyze misses of svm classifier
    #plotMissHistogram(missDict)

    # norma classifier
    #norma = NormaClassifier(xTrain,yTrain,xTest,yTest)
    #norma.classify(numDocsTrain,numDocsTest)
    
    # directed classifier
    #dirClf = Directed(xTrain,yTrain,xTest,yTest,exc,bal)
    #dirClf.classify([0,1,2,3],verb=False)

    # tune dependencies order in dirClf
    """ 
    for o in list(itertools.permutations([0,1,2,3])):
        print o
    dirClf.classify(order=list(o),verb=False)
    """

    xTrain = xTrain#[:600]
    yTrain.doc = yTrain.doc#[:600]
    yTrain.cat = yTrain.cat#[:600]
    yTrain.score = yTrain.score#[:600]
    yTrain.trueScore = yTrain.trueScore#[:600]

    # tune the classifier
    # by assigning weights to the probabilities

    # init params
    params = [[0.3,0.3,0.4],[0.3,0.3,0.4],[0.3,0.3,0.4],[0.3,0.3,0.4]]
    arr = []
    t1 = time.time()
    K=4
    portion = [[],Feature(),[],Feature()]
    portionArr = [portion for i in range(K)]
    # split the data into k parts
    for k in range(K):
        portionArr[k][0], portionArr[k][1], portionArr[k][2], portionArr[k][3] = splitData(xTrain,yTrain,K,k)
        
    # init accuracy
    maxAcc=0
    for k in range(K):
        #print "Init, K:{0} Length:{1}".format(k,len(portionArr[k][2]))
        udirClf = Undirected(portionArr[k][0],portionArr[k][1],portionArr[k][2],portionArr[k][3],params,exc,bal)
        maxAcc = udirClf.classify(trainOn='aspect',epsilon=0.0001)
    maxAcc=maxAcc/K

    """
        find the weights that maximize the accuracy
        using a greedy algorithm:
        we set the first parameter that maximizes it
        and then we set the second, and then 
        the third is the complement to 1
    """
    for cat in range(4):
        maxi = params[cat][0]
        # first parameter
        for i in [x * 0.1 for x in range(0, 11)]:
            params[cat][0] = i
            acc=0
            for k in range(K):
                #print "First param, K:{0} Length:{1}".format(k,len(portionArr[k][2]))
                udirClf = Undirected(portionArr[k][0],portionArr[k][1],portionArr[k][2],portionArr[k][3],params,exc,bal)
                acc = udirClf.classify(trainOn='aspect',epsilon=0.0001)
            acc=acc/K
            if maxAcc < acc:
                maxAcc = acc
                maxi = i
        params[cat][0] = maxi
        params[cat] = normalizeWeight(params[cat])
        maxj = params[cat][1]
        # second parameter
        for j in [x * 0.1 for x in range(0, 11)]:
            params[cat][1] = j
            acc=0
            for k in range(K):
                udirClf = Undirected(portionArr[k][0],portionArr[k][1],portionArr[k][2],portionArr[k][3],params,exc,bal)
                acc = udirClf.classify(trainOn='aspect',epsilon=0.0001)
            acc=acc/K
            if maxAcc < acc:
                maxAcc = acc
                maxj = j
        params[cat][1] = maxj
        # normalize and set third paramater
        params[cat] = normalizeWeight(params[cat])

    # tuned classifier
    udirClf = Undirected(xTrain,yTrain,xTest,yTest,params,exc,bal)
    acc = udirClf.classify(trainOn='aspect',epsilon=0.0001)
    #udirClf.classify(trainOn='all',epsilon=0.0001)
    print "Correct params: {0}".format(params)
    t2 = time.time()
    print t2-t1
