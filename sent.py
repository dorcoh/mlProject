import math
import sys
import copy
import numpy as np
import itertools

# Load classes and functions from modules
# These modules were also written by us
from DataStructures import Feature
from DataStructures import TextData
from DataStructures import Pipe
from DataStructures import printRes
from Stats import plotStats

# a global dictionary
# for categories of each paragraph
catDict = {
    u'movie': 0,
    u'extras': 1,
    u'video': 2,
    u'audio': 3
}

def multi_delete(list_, elems):
    # delete multiple elements from a list in parallel
    indexes = sorted(elems, reverse=True)
    for index in indexes:
        del list_[index]
    return list_

def svmClassifier(xTrain, yTrain, docsTrain, xTest, yTest, docsTest):
    """
        Classifies samples using svm pipeline
        That works as the following:
        Data -> CountVectorizer -> TfidfTransformer -> Classifier

        Implements two classifiers as following:
        (1)
            X - Paragraps, Y - Aspect (category)
        (2)
            X - Paragraphs, Y - Sentiment (True,False)

    """
    # predict
    p = Pipe()

    clf = p.svmpipeline.fit(xTrain, yTrain.cat)
    predicted = clf.predict(xTest)
    printRes("SVM-category",predicted,yTest.cat,testData.catCat)

    clf = p.svmpipeline.fit(xTrain, yTrain.score)
    predicted = clf.predict(xTest)
    printRes("SVM-score",predicted,yTest.score)

    clf = p.svmpipeline.fit(xTrain, yTrain.trueScore)
    predicted = clf.predict(xTest)
    
    for i in range(0, len(predicted)):
        if predicted[i] >= 5:
            predicted[i] = True
        else:
            predicted[i] = False

    printRes("SVM-**SCORE**",predicted,yTest.score)
    #printRes("SVM-true score",predicted,yTest.trueScore)"""

def normaClassifier(xTrain, yTrain, docsTrain, xTest, yTest, docsTest):
    """
        Classifier:
            X - Parargraps, Y - Aspect (category)

        Classifies samples by computing the norma (of scores)
        Between a test sample and all training samples
        And then smooth the scores using weights
        The weights are computed by how near is the vector
        to other vectors in the training set
        (scores are for example [4,5,6,9] - one for each category)

        Params:
            x,y,length (train)
            x,y,length (test)
            

        Prints results
    """
    # generate predicted - predicted samples from test set
    p = Pipe()
    clf = p.svmpipeline.fit(xTrain, yTrain.trueScore)
    predicted = clf.predict(xTest)
    # append all document's scores to vector of size 4
    # [scoreMovie, scoreExtras, scoreVideo, scoreAudio]
    # training scores
    scoresTrain = [[0,0,0,0] for i in range(docsTrain)]
    for i in range(0, len(xTrain)):
        scoresTrain[yTrain.doc[i]][catDict[yTrain.cat[i]]] = yTrain.trueScore[i]

    # testing scores (predicted)
    scoresTest = [[0,0,0,0] for i in range(docsTest)]
    for i in range(0, len(xTest)):
        scoresTest[yTest.doc[i]][catDict[yTest.cat[i]]] = predicted[i]

    # compute difference between each score vector 
    # in test and all training score vector = 
    fixScores = [[0,0,0,0] for i in range(docsTest)]
    fixedPredicted = [[0,0,0,0] for i in range(docsTest)]
    # iterating over all scores (test)
    for i in range(0, len(scoresTest)):
        # create vector of lentgh 4 from test scores (predicted)
        a = np.array(scoresTest[i])
        weightSum = 0
        tempFixScore = [0, 0, 0, 0]
        # iterate over all scores (train)
        for j in range(0, len(scoresTrain)):
            # create vector of lentgh 4 from train scores
            b = np.array(scoresTrain[j])
            # compute norma between 
            weight = np.linalg.norm(a-b)
            # compute weights for fix
            # if the norma is zero
            if weight == 0:
                # give maximum weight
                weight = (2)**3
            else:
                # else give weight: (1/norm)^3
                weight = (float(1) / weight)**3
            # sum all weights, for normalizing later
            weightSum += weight
            # apply weight for all training scores
            weightedScore = [weight*s for s in scoresTrain[j]]
            # 
            tempFixScore = [a+b for a,b in zip(tempFixScore, weightedScore)]
        #print tempFixScore
        fixScores[i] = [float(a)/weightSum for a in tempFixScore]
        #print "Last:{0} , Predicted: {1}".format(fixScores[i], scoresTest[i])
        fixedPredicted[i] = [float(0.05*a)+float(0.95*b) for a,b in zip(fixScores[i], scoresTest[i])]
        #print "Fixed: {0}".format(fixedPredicted)
        #sys.exit(2)
    
    # transform doc array to paragraph array ([[1,2,3,4]] -> [1,2,3,4])
    #print fixedPredicted
    newPredicted = []
    for i in range(0, len(fixedPredicted)):
        for j in range(0, 4):
            newPredicted.append(fixedPredicted[i][j])

    # transform predicted score to True/False
    for i in range(0, len(newPredicted)):
        if newPredicted[i] >= 5:
            newPredicted[i] = True
        else:
            newPredicted[i] = False

    clf = p.svmpipeline.fit(xTrain, yTrain.score)
    predicted = clf.predict(xTest)
    newPredicted = np.array(newPredicted)
    printRes("SVM-SmoothedScore",newPredicted,yTest.score)
    printRes("SVM-old-trueScore",predicted,yTest.score)

def groupData(x,y,numOfDocs):
    # group scores by document, e.g [6,5,4,9]

    scores = [[0,0,0,0] for i in range(numOfDocs)]
    for i in range(0, len(x)):
        scores[y.doc[i]][catDict[y.cat[i]]] = y.score[i]

    return scores

def determine(ycat,yscore,cat,value):
    """ 
        determine which samples to remove
        by choosing only categories with specified values
        Movie=False
        
        * returns 
            True if paragraph should be removed
            False otherwise
    """
    if catDict[ycat] == cat and yscore != value:
        return True
    else:
        return False

def filterData(x,y,cat,value):
    """
        filter data by assigning a value to some aspect
        e.g: Movie=True
        returns: 
            a list of indices of paragraphs "to-remove"
    """ 
    elems = set()
    # find all 'bad' docs indexes
    for i in range(0,len(x)):
        if determine(y.cat[i],y.score[i],cat,value):
            elems.add(y.doc[i])

    # find all 'bad' paragraphs indexes
    badIndexes = []
    for i in range(0, len(x)):
        if y.doc[i] not in elems:
            badIndexes.append(i)

    return badIndexes

def generateClassifiers(xTrain, yTrain,xTest,yTest,order):
    """
        This function generates 15 different classifiers
        as the following:
            1st - clf on xTrain
            2nd - clf on xTrain:A=0
            3rd - clf on xTrain:A=1
            4th - clf on xTrain:A=0,B=0
            and so on..

            returns:
                clfList - a dictionary of classifiers
                allSet - a corresponding dictionary of training sets
    """
    p = Pipe()
        
    currSet = [('-1',(xTrain, yTrain))]
    allSet = copy.deepcopy(currSet)

    translator = [False, True]

    # generate list of tuples (id, trainingSet)
    for i in order[:-1]:
        newCurrList = []
        for elem in currSet:
            for j in [0,1]:
                key = str(elem[0]) + ',' + str(i) + ':' + str(j)
                # find indexes to delete
                badIndexes = filterData(elem[1][0],elem[1][1],i,translator[j])
                newX = copy.deepcopy(elem[1][0])
                newY = copy.deepcopy(elem[1][1])
                # delete data
                multi_delete(newX, badIndexes)
                newY.removeItems(badIndexes)
                # append new data
                newCurrList.append((key,(newX,newY)))
        allSet = allSet + newCurrList
        currSet = newCurrList

    for tup in allSet:
        print tup[0], len(tup[1][0])

    # generate a list of tuples (id, classifier)
    clfList = []
    for tup in allSet:
        clf = p.svmpipeline.fit(tup[1][0], tup[1][1].score)
        clfList.append((tup[0],copy.deepcopy(clf)))
    
    return dict(clfList),dict(allSet)

def chooseSample(xTest, yTest, docId, aspect):
    """
        This function chooses a sample by document id and aspect
        params:
            xTestSet, yTestSet, documentId, aspect (0 - Movie, 1, 2, 3)
        returns:
            X -
                One paragraph (Movie/Extras/Video/Audio)
            Y -
                Feature object
    """
    x = []
    y = Feature()
    numOfAspects = 4
    # compute index
    i = docId*numOfAspects+aspect
    # get data
    x.append(xTest[i])
    y.cat.append(yTest.cat[i])
    y.score.append(yTest.score[i])
    y.trueScore.append(yTest.trueScore[i])
    y.doc.append(yTest.doc[i])

    return x,y

def predictSample(xTest,yTest,docId,clfDict,setsDict,misDict,order):
    """
        This function predicts label for a single sample (document)
        By maximizing the following equation:
        P(A,B,C,D|Sample) = P(A|Sample)*P(B|A,Sample)*P(C|A,B,Sample)*P(D|A,B,C,Sample)
        Where A,B,C,D are the aspects of each paragraph

        The function uses different classifier
        In order to find the probability for each element
        e.g (first element), P(A=0|Sample), P(A=1|Sample)

        returns:
            predicted score, e.g: [True,True,False,False]
    """
    # append all four categories (paragraphs) to x
    x = []
    for i in order:
        data,labels = chooseSample(xTest,yTest,docId,i)
        x.append(data)

    # append first probabilities
    mainKey = '-1'
    clf = clfDict[mainKey]
    probsA = clf.predict_proba(x[0])[0]
    probList = list()

    # run in 4 loops to get total 16 probabilities
    for a in xrange(0,2):
        aKey = '{0}:{1}'.format(order[0],a)
        clfKey = mainKey + str(',') + aKey
        clf = clfDict[clfKey]
        probsB = clf.predict_proba(x[1])[0]

        for i in xrange(0,2):
            bKey = '{0}:{1}'.format(order[1],i)
            clfKey = mainKey + str(',') + aKey + str(',') + bKey
            clf = clfDict[clfKey]
            probsC = clf.predict_proba(x[2])[0]
            
            for j in xrange(0,2):
                cKey = '{0}:{1}'.format(order[2],j)
                clfKey = mainKey + str(',') + aKey + str(',') + bKey + str(',') + cKey
                clf = clfDict[clfKey]
                probsD = clf.predict_proba(x[3])[0] 
                
                for k in xrange(0,2):
                    dKey = '{0}:{1}'.format(order[3],k)
                    sequenceKey = mainKey + str(',') + aKey + str(',') + bKey + str(',') + cKey + str(',') + dKey;
                    # append the probability, e.g: P(A=0,B=0,C=0,D=0|Sample)=0.02
                    probList.append((sequenceKey,probsA[a]*probsB[i]*probsC[j]*probsD[k]))

    probDict = dict(probList)
    # find the maximum probability from all possibilities
    maxKey = max(probDict.iterkeys(), key=(lambda key: probDict[key]))
    minKey = min(probDict.iterkeys(), key=(lambda key: probDict[key]))

    result = [None,None,None,None]
    # parse the maximum key to get the assignment
    # result is a vector of size 4
    for i in range(0,4):
        result[order[i]] = (bool(int(maxKey.split(',')[i+1][2])))
    # stats
    """
    trueLabel = []
    startIdx = docId*4
    for i in range(startIdx,startIdx+4):
        trueLabel.append(yTest.score[i])

    print "Predict: {0}, Prob:{1}".format(result, probDict[maxKey])
    
    trueProbKey = '-1'
    for i in range(0,4):
        s = int(trueLabel[i])
        trueProbKey = trueProbKey + str(',') + str(i) + str(':') + str(s)
    print "TrueLabel: {0}, Prob:{1}".format(trueLabel, probDict[trueProbKey])
    

    mis = (result==trueLabel)

    keys = ''
    if not mis:
        for i in range(0,4):
            if result[i] != trueLabel[i]:
                misDict[i] += 1
                keys = keys + str(i) + str(';')

    print "Mistaken: {0}, Keys:{1}".format(not mis, keys)
    print "---------------------"
    """

    return result

def superbClassifier(xTrain,yTrain,xTest,yTest,order):
    """
        This function implemenets a classifier
        Which maximizes the previous equation
        For each sample

        Prints results (accuracy, confusion matrix)
    """
    misDict = {0:0, 1:0, 2:0, 3:0}
    print "Current Order: {0}".format(order)

    # gets classifiers dict, training sets dict
    clfList, allSet = generateClassifiers(xTrain, yTrain, xTest,yTest,order)
    predScores = []
    # predict label for each doument (4 paragraphs)
    for doc in range(0, len(xTest)/4):
        docScores = predictSample(xTest,yTest,doc,clfList,allSet,misDict,order)
        for i in range(0,4):
            predScores.append(docScores[i])
    """
    print "Total misses:"
    for key in misDict:
        print "Key:{0}, Misses:{1}".format(key, misDict[key])
    """
    mainKey = '-1'
    clf = dict(clfList)[mainKey]

    # get baseline prediction (svm)
    baselinePred = clf.predict(xTest)
    baselinePred = np.array(baselinePred)
    predScores = np.array(predScores)
    # print results
    acc = printRes('superb',predScores,yTest.score)
    return acc

"""  Main function """

if __name__ == '__main__':
    trainData = TextData('train-reviews.json',balance=False)
    testData = TextData('test-reviews.json',balance=False)

    xTrain, yTrain, docsTrain = trainData.x, trainData.y, trainData.docs
    xTest, yTest, docsTest = testData.x, testData.y, testData.docs
    #svmClassifier(xTrain, yTrain, docsTrain, xTest, yTest, docsTest)
    lst = [0,1,2,3]
    perms = list(itertools.permutations(lst))
    maxacc = 0
    maxOrder = None
    for p in perms:
        acc = superbClassifier(xTrain,yTrain,xTest,yTest,list(p))
        if acc > maxacc:
            maxacc = acc
            maxOrder = list(p)

    print maxOrder,maxacc
    # assumption checking
    #plotStats(xTrain,yTrain,docsTrain,xTest,yTest,docsTest)
    
    # filtering
    """
    badIndexes = filterData(xTrain,yTrain,0,False)
    newX = multi_delete(copy.deepcopy(xTrain),badIndexes)
    newY = copy.deepcopy(yTrain)
    newY.removeItems(badIndexes)

    print len(newX), len(newY.cat)
    badIndexes = filterData(newX, newY, 1, False)
    multi_delete(newX,badIndexes)
    newY.removeItems(badIndexes)
    print len(newX), len(newY.cat)

    badIndexes = filterData(newX, newY, 2, True)
    multi_delete(newX,badIndexes)
    newY.removeItems(badIndexes)
    print len(newX), len(newY.cat)
    """

    # classifiers
    #svmClassifier(xTrain, yTrain, docsTrain, xTest, yTest, docsTest)
    #normaClassifier(xTrain, yTrain, docsTrain, xTest, yTest, docsTest)
