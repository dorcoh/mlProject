"""
    This module was used for assumption checking
    We wanted to check how the scores are dependent on the avg
    in order to implement the norma classifier
    (Described in Classifiers module)
"""

import numpy as np
import matplotlib.pyplot as plt

def printStats(x,y,docs):
    """
        Computes an histogram of variances
        of pargaraph's score in the same document.
        
        then compute the probability from this histogram
        for any variance

        In:
            x - paragraphs
            y - Feature object
            docs - number of documents in data
    """

    # computes the sum of score for each doc
    sums = [0] * docs
    numPar = [0] * docs

    for i in range(0,len(x)):
        sums[y.doc[i]] += y.trueScore[i]
        numPar[y.doc[i]] += 1

    # validity check
    if len(numPar) != len(sums):
        sys.exit("Error")

    # compute average score for each doc
    avg = [0] * docs
    for j in range(0, len(sums)):
        if numPar[j] != 4:
            print j, numPar[j]
            sys.exit("Error numPar[j] != 4")
        avg[j] = float(sums[j]) / numPar[j]


    # compute the variance for each doc
    # compute for each paragraph the diff from avg of its document
    var = [0] * docs
    diff = [0] * len(x)
    for i in range(0, len(x)):
        var[y.doc[i]] += (y.trueScore[i]-avg[y.doc[i]])**2
        diff[i] = y.trueScore[i] - avg[y.doc[i]]

    for i in range(0, len(var)):
        var[i] = float(var[i]) / (numPar[i]-1)

    histVar, binsVar = np.histogram(var, bins=40, range=(0,20), density=False)
    histDiff, binsDiff = np.histogram(diff, bins=40, range=(-10.0,10.0), density=False)

    return histVar,binsVar,histDiff,binsDiff

def outVar(x,y):
    # computes the avg of all scores
    sumScore = 0
    for i in range(0,len(x)):
        sumScore += y.trueScore[i]
    
    avgScore = sumScore / len(x)

    # compute variance
    var = 0
    for i in range(0, len(x)):
        var += (y.trueScore[i]-avgScore)**2

    var = float(var)/(len(x)-1)

    return var

def plotStats(xTrain,yTrain,docsTrain,xTest,yTest,docsTest):
    # plot stats
    #f, axarr = plt.subplots(2)

    # train data
    histVar, binsVar, histDiff, binsDiff = printStats(xTrain,yTrain,docsTrain)

    # plot histogram
    plt.bar(binsVar[:-1],histVar,width=0.5)
    plt.title("Training set - scores variance")
    plt.xlabel("Score's variance")
    plt.ylabel("Occurences")
    plt.show()

def plotMissHistogram(missDict):
    plt.bar(missDict.keys(),missDict.values(),width=1,color="g")
    plt.title("Miss histogram (by rating)")
    plt.xlabel("Rating")
    plt.ylabel("Misses")
    plt.show()

def plotRatioHistogram(ratioDict):
    keys = tuple(ratioDict.keys())
    vals = ratioDict.values()
    yVal = np.asarray(vals)
    yPos = np.arange(len(keys))

    plt.barh(yPos,yVal,align='center',alpha=0.4)
    plt.yticks(yPos,keys)
    plt.xlabel("Perecent")
    plt.title("Positive-Negative Ratio in training set")

    plt.show()