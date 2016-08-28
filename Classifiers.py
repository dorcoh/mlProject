"""
    Module for implementation of the base classifiers
"""
import copy
import numpy as np
from sklearn import metrics
from DataStructures import Feature
from DataStructures import TextData
from DataStructures import Pipe
from DataStructures import multi_delete
from DataStructures import catDict

class Classifier:
    """
        A base class for our custom classifiers
    """

    def __init__(self, xTrain, yTrain, xTest, yTest):
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xTest = xTest
        self.yTest = yTest
        self.numOfCat = 4

    def filterData(self, x, y, cat, value, requiredCat = None):
        """
            filter data by assigning a value to some aspect
            e.g: Movie=True
            optional: requiredCat - to specify which category to not remove
            returns: 
                a list of indices of paragraphs "to-remove"
        """
        elems = set()
        # find all bad docs indexes
        # filter only cat=value
        for i in range(0, len(x)):
            if self.determine(y.cat[i], y.score[i], cat, value):
                elems.add(y.doc[i])

        # find the bad pargraphs (whole docs)
        badIndexes = []
        for i in range(0, len(x)):
            if y.doc[i] in elems:
                badIndexes.append(i)

        if requiredCat != None:
            for i in range(0, len(x)):
                if y.doc[i] in elems:
                    continue
                if catDict[y.cat[i]] != requiredCat:
                    badIndexes.append(i)

        return badIndexes

    def determine(self, yCat, yScore, cat, value):
        """ 
            determine which samples to remove
            by choosing only categories with specified values
            e.g Movie=False
            
            * returns 
                True if paragraph should be removed
                False otherwise
        """
        if catDict[yCat] == cat and yScore != value:
            return True
        else:
            return False

    def chooseSample(self, xTest, yTest, docId, aspect):
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
        i = docId * self.numOfCat + aspect
        x.append(xTest[i])
        y.cat.append(yTest.cat[i])
        y.score.append(yTest.score[i])
        y.trueScore.append(yTest.trueScore[i])
        y.doc.append(yTest.doc[i])
        return (x,y)

    def printRes(self, title, predicted, yTest, cats = None):
        """
            print results
            accuracy, confusion, report
        
            returns accuracy
        """
        print '______________________'
        print title
        print '______________________'
        acc = np.mean(predicted == yTest)
        print 'Acc={0}'.format(acc)
        print 'Confusion Matrix:'
        print metrics.confusion_matrix(yTest, predicted)
        if cats != None:
            print 'Classification Report:'
            print metrics.classification_report(yTest, predicted, target_names=cats)
        print
        return acc


class Directed(Classifier):

    def classify(self, order, verb = False):
        """
            This function implemenets a classifier
            Which maximizes this equation for all samples:
            P(A,B,C,D|Sample) = P(A|Sample)*P(B|A,Sample)*P(C|A,B,Sample)*P(D|A,B,C,Sample)
            Where A,B,C,D are the aspects of each document
        
            In:
                order: list of the order of the dependencies
                       e.g [0,1,2,3]
                verb:  show more details for each prediction
            Out:
        
            Prints results (accuracy, confusion matrix)
        """
        xTrain, yTrain = self.xTrain, self.yTrain
        xTest, yTest = self.xTest, self.yTest
        if verb:
            missDict = {0: 0,1:0,2:0,3:0}
        else:
            missDict = None
        clfList, allSet = self.generateClassifiers(xTrain, yTrain, xTest, yTest, order)
        predScores = []
        for docId in range(0, len(xTest) / self.numOfCat):
            docScores = self.predictSample(xTest, yTest, docId, clfList, allSet, missDict, order)
            for i in range(0, 4):
                predScores.append(docScores[i])

        mainKey = '-1'
        clf = dict(clfList)[mainKey]
        baselinePred = clf.predict(xTest)
        baselinePred = np.array(baselinePred)
        predScores = np.array(predScores)
        acc = self.printRes('One-direction-dependency', predScores, yTest.score)
        if verb:
            print 'Total misses:'
            for key in missDict:
                print 'Key:{0}, Misses:{1}'.format(key, missDict[key])

    def predictSample(self, xTest, yTest, docId, clfDict, setsDict, missDict, order):
        """
            This function predicts label for a single sample (document)
        
            The function uses different classifier
            In order to find the probability for each element
            e.g (first element), P(A=0|Sample), P(A=1|Sample)
        
            returns:
                predicted score, e.g: [True,True,False,False]
        """
        x = []
        for i in order:
            data, labels = self.chooseSample(xTest, yTest, docId, i)
            x.append(data)

        mainKey = '-1'
        clf = clfDict[mainKey]
        probsA = clf.predict_proba(x[0])[0]
        probList = list()
        for a in xrange(0, 2):
            aKey = '{0}:{1}'.format(order[0], a)
            clfKey = mainKey + str(',') + aKey
            clf = clfDict[clfKey]
            probsB = clf.predict_proba(x[1])[0]
            for i in xrange(0, 2):
                bKey = '{0}:{1}'.format(order[1], i)
                clfKey = mainKey + str(',') + aKey + str(',') + bKey
                clf = clfDict[clfKey]
                probsC = clf.predict_proba(x[2])[0]
                for j in xrange(0, 2):
                    cKey = '{0}:{1}'.format(order[2], j)
                    clfKey = mainKey + str(',') + aKey + str(',') + bKey + str(',') + cKey
                    clf = clfDict[clfKey]
                    probsD = clf.predict_proba(x[3])[0]
                    for k in xrange(0, 2):
                        dKey = '{0}:{1}'.format(order[3], k)
                        sequenceKey = mainKey + str(',') + aKey + str(',') + bKey + str(',') + cKey + str(',') + dKey
                        probList.append((sequenceKey, probsA[a] * probsB[i] * probsC[j] * probsD[k]))

        probDict = dict(probList)
        maxKey = max(probDict.iterkeys(), key=lambda key: probDict[key])
        result = [None for i in range(self.numOfCat)]
        for i in range(self.numOfCat):
            result[order[i]] = bool(int(maxKey.split(',')[i + 1][2]))

        if missDict:
            self.showStats(yTest, probDict, result, missDict, docId, maxKey)
        return result

    def generateClassifiers(self, xTrain, yTrain, xTest, yTest, order):
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
        currSet = [('-1', (xTrain, yTrain))]
        allSet = copy.deepcopy(currSet)
        translator = [False, True]
        for i in order[:-1]:
            newCurrList = []
            for elem in currSet:
                for j in [0, 1]:
                    key = str(elem[0]) + ',' + str(i) + ':' + str(j)
                    badIndexes = self.filterData(elem[1][0], elem[1][1], i, translator[j])
                    newX = copy.deepcopy(elem[1][0])
                    newY = copy.deepcopy(elem[1][1])
                    multi_delete(newX, badIndexes)
                    newY.removeItems(badIndexes)
                    newCurrList.append((key, (newX, newY)))

            allSet = allSet + newCurrList
            currSet = newCurrList

        clfList = []
        for tup in allSet:
            clf = p.svmpipeline.fit(tup[1][0], tup[1][1].score)
            clfList.append((tup[0], copy.deepcopy(clf)))

        return (dict(clfList), dict(allSet))

    def showStats(self, yTest, probDict, result, missDict, docId, maxKey):
        trueLabel = []
        startIdx = docId * self.numOfCat
        for i in range(startIdx, startIdx + self.numOfCat):
            trueLabel.append(yTest.score[i])

        print 'Predict: {0}, Prob:{1}'.format(result, probDict[maxKey])
        trueProbKey = '-1'
        for i in range(0, 4):
            s = int(trueLabel[i])
            trueProbKey = trueProbKey + str(',') + str(i) + str(':') + str(s)

        print 'TrueLabel: {0}, Prob:{1}'.format(trueLabel, probDict[trueProbKey])
        miss = result != trueLabel
        keys = ''
        if miss:
            for i in range(0, 4):
                if result[i] != trueLabel[i]:
                    missDict[i] += 1
                    keys = keys + str(i) + str(';')

        print 'Mistaken: {0}, Keys:{1}'.format(miss, keys)
        print '---------------------'


class Undirected(Classifier):

    def classify(self, trainOn, epsilon = 0.0001):
        """
            This functions classifies all samples
        
            For each aspect,
            it iterates over this equation until P_t+1-P_t <= epsilon:
        
            P_t+1(A|v) * 3 = 
            P_t(B=0|v)*P_b0(A|v)+P_t(B=1|v)*P_b1(A|v)+
            P_t(C=0|v)*P_c0(A|v)+P_t(C=1|v)*P_c1(A|v)+
            P_t(D=0|v)*P_d0(A|v)+P_t(D=1|v)*P_d1(A|v)
        
            In:
                trainOn - can get 'all' or 'aspect'
                          it defines on which samples the
                          inner classifiers would train
                epsilon - defines when to stop iterating
        
            Prints results (accuracy, confusion matrix)
        """
        xTrain, yTrain = self.xTrain, self.yTrain
        xTest, yTest = self.xTest, self.yTest
        numOfCat = self.numOfCat
        clfList, allSet = self.generateClassifiers(xTrain, yTrain, xTest, yTest, trainOn)
        predScores = []
        for doc in range(0, len(xTest) / numOfCat):
            docScores = self.predictSample(xTest, yTest, doc, clfList, allSet, trainOn, epsilon)
            for i in range(0, numOfCat):
                predScores.append(docScores[i])

        mainKey = '-1'
        clf = dict(clfList)[mainKey]
        titles = {'all': 'Pairwise-trained by all aspects',
         'aspect': 'Pairwise-trained by aspect'}
        self.printRes(titles[trainOn], np.array(predScores), yTest.score)

    def predictSample(self,xTest,yTest,docId,clfDict,setsDict,trainOn,epsilon):
        """
            This function predicts label for a single sample (document)
            By iterating on the following equation (for each aspect)
            P_t+1(A|v) * 3 = 
            P_t(B=0|v)*P_b0(A|v)+P_t(B=1|v)*P_b1(A|v)+
            P_t(C=0|v)*P_c0(A|v)+P_t(C=1|v)*P_c1(A|v)+
            P_t(D=0|v)*P_d0(A|v)+P_t(D=1|v)*P_d1(A|v)
        """

        # append all paragraphs of document to x
        order = [0,1,2,3]
        x = []
        for i in order:
            data,labels = self.chooseSample(xTest,yTest,docId,i)
            x.append(data) 

        mainKey = '-1'
        clf = clfDict[mainKey]

        probs = [0 for i in range(self.numOfCat)]
        for i in range(self.numOfCat):
            probs[i] = clf.predict_proba(x[i])[0]

        # create permutations
        perms = []
        for i in range(len(order)):
            temp = copy.deepcopy(order)
            temp.remove(i)
            perms.append(temp)

        P = [0 for i in range(self.numOfCat)]

        # iterate over equation
        while (1):
            for cat in range(self.numOfCat):
                P[cat] = 0
                for j in perms[cat]:
                    for bin in [0,1]:
                        if trainOn == 'all':
                            key = '-1,{0}:{1}'.format(j,bin)
                        elif trainOn == 'aspect':
                            key = '-1,{0}:{1},T:{2}'.format(j,bin,cat)
                        # sum
                        P[cat] += probs[j][bin]*clfDict[key].predict_proba(x[cat])
                # normalize
                P[cat] = P[cat] / 3
            
            # compute norma
            a = [probs[i][0] for i in range(self.numOfCat)]
            b = [P[i][0][0] for i in range(self.numOfCat)]
            a = np.array(a)
            b = np.array(b)
            
            # save probs
            for i in range(self.numOfCat):
                probs[i] = P[i][0]

            # stop loop if norma is lower than epsilon
            if np.linalg.norm(a-b) < epsilon:
                break

        result = [None for i in range(self.numOfCat)]

        # get result
        prob = [probs[i] for i in range(self.numOfCat)]
        for i in range(0,4):
            if probs[i][0] > probs[i][1]:
                result[i] = False
            else:
                result[i] = True

        return result

    def generateClassifiers(self, xTrain, yTrain, xTest, yTest, trainOn):
        """
            Generates different classifiers:
            1st - clf on xTrain
        
            In case of trainOn = 'all':
            2nd - clf on xTrain:A=0
            3rd - clf on xTrain:A=1
            4th - clf on xTrain:B=0
            and so on... (total 1+4*2)
        
            In case of trainOn = 'aspect'
            2nd - clf on xTrain:A=0,TrainOn:A
            3rd - clf on xTrain:A=1,TrainOn:A
            4th - clf on xTrain:B=0,TrainOn:A
            and so on... (total 1+4*6)
        
            returns:
                clfList - a dict of classifiers
                allSet - a corresponding dict of training sets
        
        """
        p = Pipe()
        baseSet = [('-1', (xTrain, yTrain))]
        allSet = copy.deepcopy(baseSet)
        self.genTuples(xTrain, yTrain, allSet, trainOn)
        clfList = []
        for tup in allSet:
            clf = p.svmpipeline.fit(tup[1][0], tup[1][1].score)
            clfList.append((tup[0], copy.deepcopy(clf)))

        return (dict(clfList), dict(allSet))

    def genTuples(self, xTrain, yTrain, allSet, trainOn):
        """
            Generates different tuples of (id,training set)
            and corresponding tuple (id, clf)
            and appends them to dictionaries
        """
        translator = [False, True]
        if trainOn == 'all':
            for cat in range(0, self.numOfCat):
                for value in range(0, 2):
                    key = str('-1,') + '{0}:{1}'.format(cat, value)
                    badIndexes = self.filterData(xTrain, yTrain, cat, translator[value])
                    newX = copy.deepcopy(xTrain)
                    newY = copy.deepcopy(yTrain)
                    multi_delete(newX, badIndexes)
                    newY.removeItems(badIndexes)
                    allSet.append((key, (newX, newY)))

        elif trainOn == 'aspect':
            order = []
            for i in range(self.numOfCat):
                curOrder = [ a for a in range(self.numOfCat) if a != i ]
                order.append(curOrder)

            for cat in range(0, self.numOfCat):
                for currCat in order[cat]:
                    for value in range(0, 2):
                        key = '-1,' + '{0}:{1},'.format(currCat, value) + 'T:{0}'.format(cat)
                        badIndexes = self.filterData(xTrain, yTrain, currCat, translator[value], cat)
                        newX = copy.deepcopy(xTrain)
                        newY = copy.deepcopy(yTrain)
                        multi_delete(newX, badIndexes)
                        newY.removeItems(badIndexes)
                        allSet.append((key, (newX, newY)))


class NormaClassifier(Classifier):

    def classify(self, docsTrain, docsTest):
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
        xTrain, yTrain = self.xTrain, self.yTrain
        xTest, yTest = self.xTest, self.yTest
        p = Pipe()
        clf = p.svmpipeline.fit(xTrain, yTrain.trueScore)
        predicted = clf.predict(xTest)
        scoresTrain = [ [0,0,0,0] for i in range(docsTrain) ]
        for i in range(0, len(xTrain)):
            scoresTrain[yTrain.doc[i]][catDict[yTrain.cat[i]]] = yTrain.trueScore[i]

        scoresTest = [ [0,0,0,0] for i in range(docsTest) ]
        for i in range(0, len(xTest)):
            scoresTest[yTest.doc[i]][catDict[yTest.cat[i]]] = predicted[i]

        fixScores = [ [0,0,0,0] for i in range(docsTest) ]
        fixedPredicted = [ [0,0,0,0] for i in range(docsTest) ]
        for i in range(0, len(scoresTest)):
            a = np.array(scoresTest[i])
            weightSum = 0
            tempFixScore = [0,0,0,0]
            for j in range(0, len(scoresTrain)):
                b = np.array(scoresTrain[j])
                weight = np.linalg.norm(a - b)
                if weight == 0:
                    weight = 8
                else:
                    weight = (float(1) / weight) ** 3
                weightSum += weight
                weightedScore = [ weight * s for s in scoresTrain[j] ]
                tempFixScore = [ a + b for a, b in zip(tempFixScore, weightedScore) ]

            fixScores[i] = [ float(a) / weightSum for a in tempFixScore ]
            fixedPredicted[i] = [ float(0.05 * a) + float(0.95 * b) for a, b in zip(fixScores[i], scoresTest[i]) ]

        newPredicted = []
        for i in range(0, len(fixedPredicted)):
            for j in range(0, 4):
                newPredicted.append(fixedPredicted[i][j])

        for i in range(0, len(newPredicted)):
            if newPredicted[i] >= 5:
                newPredicted[i] = True
            else:
                newPredicted[i] = False

        newPredicted = np.array(newPredicted)
        self.printRes('Norma CLF: SVM-SmoothedScore', newPredicted, yTest.score)


class SvmClassifier(Classifier):

    def classify(self, catList):
        """
            Classifies samples using svm pipeline
            That works as the following:
            Data -> CountVectorizer -> TfidfTransformer -> Classifier
        
            Implements three classifiers as following:
            (1)
                X - Paragraps, Y - Aspect (category)
            (2)
                Train on 
                X - Paragraph, Y - Sentiment (True,False)
                Output Sentiment
            (3)
                Train on 
                X - Paragraph, Y - Score (1-10)
                Output sentiment
        """
        xTrain, yTrain = self.xTrain, self.yTrain
        xTest, yTest = self.xTest, self.yTest
        p = Pipe()
        
        clf = p.svmpipeline.fit(xTrain, yTrain.cat)
        predicted = clf.predict(xTest)
        self.printRes('SVM-category', predicted, yTest.cat, catList)
        
        clf = p.svmpipeline.fit(xTrain, yTrain.score)
        predicted = clf.predict(xTest)
        self.printRes('SVM-score (Trained on score)', predicted, yTest.score)
        
        clf = p.svmpipeline.fit(xTrain, yTrain.trueScore)
        predicted = clf.predict(xTest)
        for i in range(0, len(predicted)):
            if predicted[i] >= 5:
                predicted[i] = True
            else:
                predicted[i] = False

        self.printRes('SVM-score (Trained on rating: 1-10)', predicted, yTest.score)
