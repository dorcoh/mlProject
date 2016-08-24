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
                key = 'T:{0},'.format(cat) + '{0}:{1}'.format(currCat,value)
                # find indexes to delete
                badIndexes = filterData(xTrain,yTrain,cat,translator[value])
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
        print tup[0]
        clf = p.svmpipeline.fit(tup[1][0],tup[1][1].score)
        clfList.append((tup[0],copy.deepcopy(clf)))

    return dict(clfList),dict(allSet)

"""  Main function """

if __name__ == '__main__':

    trainData = TextData('train-reviews.json',balance=False)
    testData = TextData('test-reviews.json',balance=False)

    xTrain, yTrain, docsTrain = trainData.x, trainData.y, trainData.docs
    xTest, yTest, docsTest = testData.x, testData.y, testData.docs
    
    #genClfsNew(xTrain,yTrain,xTest,yTest)

    # filtering check
    newX  = copy.deepcopy(xTrain)
    newY = copy.deepcopy(yTrain)

    print len(newX), len(newY.cat)
    badIndexes = filterData(newX, newY, 1, False)
    multi_delete(newX,badIndexes)
    newY.removeItems(badIndexes)
    print len(newX), len(newY.cat)

    for i in range(0,len(newY.cat)):
        print "Cat:{0},Val:{1}".format(catDict[newY.cat[i]],int(newY.score[i]))
    # classifiers check
    # svmClassifier(xTrain, yTrain, docsTrain, xTest, yTest, docsTest)
    # normaClassifier(xTrain, yTrain, docsTrain, xTest, yTest, docsTest)
    # sClassifier(xTrain,yTrain,xTest,yTest)
    # superbClassifier(xTrain,yTrain,xTest,yTest,[0,1,2,3])