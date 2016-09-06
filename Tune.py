"""
    This module was used to tune our custom classifiers
"""

import itertools
import math
import time

from DataStructures import Feature
from Classifiers import Undirected

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

def tuneUndirected(xTrain,yTrain,initParams,exc,bal,foldTo=4):
    """
        #find the weights that maximize the accuracy
        #using a greedy algorithm:
        #we set the first parameter that maximizes it
        #and then we set the second, and then 
        #the third is the complement to 1
    """
    params = initParams
    arr = []
    t1 = time.time()
    K=foldTo
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
    
    return params

def tuneDirected(xTrain,yTrain,xTest,yTest,exc,bal):
    """
        Tune directed classifier
        Find the dependency order that
        maximizes the accuracy
    """
    dirClf = Directed(xTrain,yTrain,xTest,yTest,exc,bal)
    maxAcc = 0
    maxOrder = [0,1,2,3]
    for o in list(itertools.permutations([0,1,2,3])):
        print o
        acc = dirClf.classify(order=list(o),verb=False)
        if acc > maxAcc:
            maxAcc = acc
            maxOrder = o

    return o