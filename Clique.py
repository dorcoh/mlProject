import copy
import numpy as np

from DataStructures import Feature
from DataStructures import TextData
from DataStructures import Pipe
from DataStructures import multi_delete
from DataStructures import printRes

catDict = {
    u'movie': 0,
    u'extras': 1,
    u'video': 2,
    u'audio': 3
}

def determine(ycat,yscore,cat,value):
    """ 
        determine which samples to remove
        by choosing only categories with specified values
        e.g Movie=False
        
        * returns 
            True if paragraph should be removed
            False otherwise
    """
    if catDict[ycat] == cat and yscore != value:
        return True
    else:
        return False

def filterData(x,y,cat,value,requiredCat=None):
    """
        filter data by assigning a value to some aspect
        e.g: Movie=True
        optional: requiredCat - to specify which category to remove
        returns: 
            a list of indices of paragraphs "to-remove"
    """ 
    elems = set()
    # find all 'bad' docs indexes
    # filter only by cat=value
    for i in range(0,len(x)):
        # if paragraph should be removed
        if determine(y.cat[i],y.score[i],cat,value):
            # append doc num of 'bad' docs
            elems.add(y.doc[i])
    # if any 'bad' pargraph - remove whole document
    badIndexes = []
    for i in range(0, len(x)):
        if y.doc[i] in elems:
            badIndexes.append(i)
    
    # optional param - filter documents by category
    if requiredCat != None:
        for i in range(0, len(x)):
            # skip documents that already marked 'remove'
            if y.doc[i] in elems:
                continue
            # mark 'remove' the other aspects in left documents
            if catDict[y.cat[i]] != requiredCat:
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

    # generate a list of tuples (id, classifier)
    clfList = []
    for tup in allSet:
        clf = p.svmpipeline.fit(tup[1][0], tup[1][1].score)
        clfList.append((tup[0],copy.deepcopy(clf)))
    
    return dict(clfList),dict(allSet)

def genClfs(xTrain,yTrain,xTest,yTest):
    """
        This function generates 9 different classifiers:
        1st - clf on xTrain
        2nd - clf on xTrain:A=0
        3rd - clf on xTrain:A=1
        4th - clf on xTrain:B=0
        and so on... (A=0,A=1,B=0,B=1,C=0,C=1,D=0,D=1)

        returns:
            clfList - a dict of classifiers
            allSet - a corresponding dict of training sets

    """

    p = Pipe()
    baseSet = [('-1',(xTrain, yTrain))]
    allSet = copy.deepcopy(baseSet)
    translator = [False, True]

    # generate list of tuples (id, training set)
    numOfAspects = 4
    for cat in range(0,numOfAspects):
        for value in range(0,2):
            key = str('-1,') + '{0}:{1}'.format(cat,value)
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
        clf = p.svmpipeline.fit(tup[1][0],tup[1][1].score)
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

def predSamp(xTest,yTest,docId,clfDict,setsDict):
    """
        This function predicts label for a single sample (document)
        By iterating on the following equation (for each aspect)

        P_t+1(A|v) * 3 = 
        P_t(B=0|v)*P_b0(A|v)+P_t(B=1|v)*P_b1(A|v)+
        P_t(C=0|v)*P_c0(A|v)+P_t(C=1|v)*P_c1(A|v)+
        P_t(D=0|v)*P_d0(A|v)+P_t(D=1|v)*P_d1(A|v)
    """
    # append all four paragraphs to x
    order = [0,1,2,3]
    x = []
    for i in order:
        data,labels = chooseSample(xTest,yTest,docId,i)
        x.append(data)

    mainKey = '-1'
    clf = clfDict[mainKey]

    # compute P_0(A|v[0]),P_0(B|v[1]),P_0(C|v[2]),P_0(D|v[3])
    probsA = clf.predict_proba(x[0])[0]
    probsB = clf.predict_proba(x[1])[0]
    probsC = clf.predict_proba(x[2])[0]
    probsD = clf.predict_proba(x[3])[0]

    epsilon = 0.0001
    while (1):
        Pa = (probsB[0]*clfDict['-1,1:0'].predict_proba(x[0])+
              probsB[1]*clfDict['-1,1:1'].predict_proba(x[0])+
              probsC[0]*clfDict['-1,2:0'].predict_proba(x[0])+
              probsC[1]*clfDict['-1,2:1'].predict_proba(x[0])+
              probsD[0]*clfDict['-1,3:0'].predict_proba(x[0])+
              probsD[1]*clfDict['-1,3:1'].predict_proba(x[0]))/3

        Pb = (probsA[0]*clfDict['-1,0:0'].predict_proba(x[1])+
              probsA[1]*clfDict['-1,0:1'].predict_proba(x[1])+
              probsC[0]*clfDict['-1,2:0'].predict_proba(x[1])+
              probsC[1]*clfDict['-1,2:1'].predict_proba(x[1])+
              probsD[0]*clfDict['-1,3:0'].predict_proba(x[1])+
              probsD[1]*clfDict['-1,3:1'].predict_proba(x[1]))/3

        Pc = (probsA[0]*clfDict['-1,0:0'].predict_proba(x[2])+
              probsA[1]*clfDict['-1,0:1'].predict_proba(x[2])+
              probsB[0]*clfDict['-1,1:0'].predict_proba(x[2])+
              probsB[1]*clfDict['-1,1:1'].predict_proba(x[2])+
              probsD[0]*clfDict['-1,3:0'].predict_proba(x[2])+
              probsD[1]*clfDict['-1,3:1'].predict_proba(x[2]))/3

        Pd = (probsA[0]*clfDict['-1,0:0'].predict_proba(x[3])+
              probsA[1]*clfDict['-1,0:1'].predict_proba(x[3])+
              probsB[0]*clfDict['-1,1:0'].predict_proba(x[3])+
              probsB[1]*clfDict['-1,1:1'].predict_proba(x[3])+
              probsC[0]*clfDict['-1,2:0'].predict_proba(x[3])+
              probsC[1]*clfDict['-1,2:1'].predict_proba(x[3]))/3
        
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
    
    # tune
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

def sClassifier(xTrain,yTrain,xTest,yTest):
    """
        first compute by standard classifier
        where v = sample (doc)
        P_0(A=0|v[0]),P_0(A=1|v[0])
        P_0(B=0|v[1]),P_0(B=1|v[1])
        P_0(C=0|v[2]),P_0(C=1|v[2])
        P_0(D=0|v[3]),P_0(D=1|v[3])

        then create 8 classifiers from the following train data:
        A=0,A=1,B=0,B=1,C=0,C=1,D=0,D=1

        compute
        P_b0(A|v),P_b1(A|v)
        P_c0(A|v),P_c1(A|v)
        P_d0(A|v),P_d1(A|v)

        then iterate over this until P_t+1-P_t <= epsilon:
        P_t+1(A|v) * 3 = 
        P_t(B=0|v)*P_b0(A|v)+P_t(B=1|v)*P_b1(A|v)+
        P_t(C=0|v)*P_c0(A|v)+P_t(C=1|v)*P_c1(A|v)+
        P_t(D=0|v)*P_d0(A|v)+P_t(D=1|v)*P_d1(A|v)
    """
    clfList, allSet = genClfs(xTrain,yTrain,xTest,yTest)
    predScores = []

    for doc in range(0, len(xTest)/4):
        docScores = predSamp(xTest,yTest,doc,clfList,allSet)
        for i in range(0,4):
            predScores.append(docScores[i])

    mainKey = '-1'
    clf = dict(clfList)[mainKey]

    printRes('Pairwise-trained by all aspects', np.array(predScores), yTest.score)

def superbClassifier(xTrain,yTrain,xTest,yTest,order):
    """
        This function implemenets a classifier
        Which maximizes the previous equation
        For each sample

        Prints results (accuracy, confusion matrix)
    """
    misDict = {0:0, 1:0, 2:0, 3:0}

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
    acc = printRes('One-direction-dependency',predScores,yTest.score)
    return acc
