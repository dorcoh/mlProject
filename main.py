import math
import sys
import copy
import numpy as np
import itertools

# Load classes and functions from our modules
from DataStructures import Feature
from DataStructures import TextData
from DataStructures import Pipe
from DataStructures import printRes
from DataStructures import multi_delete
from DataStructures import normaClassifier,svmClassifier
from Analyze import plotStats
from Clique import sClassifier, superbClassifier
from Clique import filterData
from Clique import chooseSample


# a global dictionary
# for categories of each paragraph
catDict = {
    u'movie': 0,
    u'extras': 1,
    u'video': 2,
    u'audio': 3
}

def genClfsNew(xTrain,yTrain,xTest,yTest):
    """
        This function generates 25 different classifiers:
        1 standard clf, and 6*4 clfs for each aspect
        where every one of them is trained on the aspect only

        1st - clf on xTrain

        2nd - clf on xTrain(Aspect=A):A=0
        3rd - clf on xTrain(Aspect=A):A=1
        4th - clf on xTrain(Aspect=A):B=0
        ...

        returns:
            clfList - a dict of classifiers
            allSet - a corresponding dict of training sets

    """

    p = Pipe()
    baseSet = [('-1',(xTrain, yTrain))]
    allSet = copy.deepcopy(baseSet)
    translator = [False, True]

    # order of needed classifiers for each aspect
    # A - 0, B - 1, C - 2, D - 3
    order = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]
    # generate list of tuples (id, training set)
    
    numOfAspects = 4
    for cat in range(0,numOfAspects):
        for currCat in order[cat]:
            for value in range(0,2):
                key = '-1,' + '{0}:{1},'.format(currCat,value) + 'T:{0}'.format(cat)
                # find indexes to delete
                badIndexes = filterData(xTrain,yTrain,currCat,translator[value],cat)
                newX = copy.deepcopy(xTrain)
                newY = copy.deepcopy(yTrain)
                # delete data
                multi_delete(newX,badIndexes)
                newY.removeItems(badIndexes)
                # append new training set
                allSet.append((key,(newX,newY)))

    clfList = []
    # train classifiers
    for tup in allSet:
        clf = p.svmpipeline.fit(tup[1][0],tup[1][1].score)
        clfList.append((tup[0],copy.deepcopy(clf)))

    return dict(clfList),dict(allSet)

def preSample(xTest,yTest,docId,clfDict,setsDict):
    order = [0,1,2,3]
    x = []
    for i in order:
        data,labels = chooseSample(xTest,yTest,docId,i)
        x.append(data)

    mainKey = '-1'
    clf = clfDict[mainKey]

    probsA = clf.predict_proba(x[0])[0]
    probsB = clf.predict_proba(x[1])[0]
    probsC = clf.predict_proba(x[2])[0]
    probsD = clf.predict_proba(x[3])[0]

    epsilon = 0.0001
    while (1):
        Pa = (probsB[0]*clfDict['-1,1:0,T:0'].predict_proba(x[0])+
              probsB[1]*clfDict['-1,1:1,T:0'].predict_proba(x[0])+
              probsC[0]*clfDict['-1,2:0,T:0'].predict_proba(x[0])+
              probsC[1]*clfDict['-1,2:1,T:0'].predict_proba(x[0])+
              probsD[0]*clfDict['-1,3:0,T:0'].predict_proba(x[0])+
              probsD[1]*clfDict['-1,3:1,T:0'].predict_proba(x[0]))/3

        Pb = (probsA[0]*clfDict['-1,0:0,T:1'].predict_proba(x[1])+
              probsA[1]*clfDict['-1,0:1,T:1'].predict_proba(x[1])+
              probsC[0]*clfDict['-1,2:0,T:1'].predict_proba(x[1])+
              probsC[1]*clfDict['-1,2:1,T:1'].predict_proba(x[1])+
              probsD[0]*clfDict['-1,3:0,T:1'].predict_proba(x[1])+
              probsD[1]*clfDict['-1,3:1,T:1'].predict_proba(x[1]))/3

        Pc = (probsA[0]*clfDict['-1,0:0,T:2'].predict_proba(x[2])+
              probsA[1]*clfDict['-1,0:1,T:2'].predict_proba(x[2])+
              probsB[0]*clfDict['-1,1:0,T:2'].predict_proba(x[2])+
              probsB[1]*clfDict['-1,1:1,T:2'].predict_proba(x[2])+
              probsD[0]*clfDict['-1,3:0,T:2'].predict_proba(x[2])+
              probsD[1]*clfDict['-1,3:1,T:2'].predict_proba(x[2]))/3

        Pd = (probsA[0]*clfDict['-1,0:0,T:3'].predict_proba(x[3])+
              probsA[1]*clfDict['-1,0:1,T:3'].predict_proba(x[3])+
              probsB[0]*clfDict['-1,1:0,T:3'].predict_proba(x[3])+
              probsB[1]*clfDict['-1,1:1,T:3'].predict_proba(x[3])+
              probsC[0]*clfDict['-1,2:0,T:3'].predict_proba(x[3])+
              probsC[1]*clfDict['-1,2:1,T:3'].predict_proba(x[3]))/3
        
        # compute norma
        a = [probsA[0],probsB[0],probsC[0],probsD[0]]
        b = [Pa[0][0], Pb[0][0], Pc[0][0], Pd[0][0]]
        a = np.array(a)
        b = np.array(b)

        # save probs
        probsA, probsB, probsC, probsD = Pa[0],Pb[0],Pc[0],Pd[0] 

        # while()
        if np.linalg.norm(a-b) < 0.0001:
            break

    result = [None,None,None,None]

    # get result
    probs = [probsA,probsB,probsC,probsD]
    for i in range(0,4):
        if probs[i][0] > probs[i][1]:
            result[i] = False
        else:
            result[i] = True

    return result    

def sClfAspect(xTrain,yTrain,xTest,yTest):
    clfList, allSet = genClfsNew(xTrain,yTrain,xTest,yTest)
    predScores = []

    for doc in range(0, len(xTest)/4):
        docScores = preSample(xTest,yTest,doc,clfList,allSet)
        for i in range(0,4):
            predScores.append(docScores[i])

    mainKey = '-1'
    clf = dict(clfList)[mainKey]
    predScores = np.array(predScores)
    printRes('Pairwise-trained by aspect', predScores, yTest.score)    

"""  Main function """

if __name__ == '__main__':

    trainData = TextData('train-reviews.json',balance=False)
    testData = TextData('test-reviews.json',balance=False)

    xTrain, yTrain, docsTrain = trainData.x, trainData.y, trainData.docs
    xTest, yTest, docsTest = testData.x, testData.y, testData.docs

    # classifiers check
    svmClassifier(xTrain, yTrain, docsTrain, xTest, yTest, docsTest, testData.catCat)
    # normaClassifier(xTrain, yTrain, docsTrain, xTest, yTest, docsTest)
    sClassifier(xTrain,yTrain,xTest,yTest)
    superbClassifier(xTrain,yTrain,xTest,yTest,[0,1,2,3])
    sClfAspect(xTrain,yTrain,xTest,yTest)