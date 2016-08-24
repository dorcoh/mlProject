"""
    Module for all of our needed classes and functions
"""
import json
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import numpy as np

def multi_delete(list_, elems):
    # delete multiple elements from a list in parallel
    indexes = sorted(elems, reverse=True)
    for index in indexes:
        del list_[index]
    return list_

class Feature:
    """
        a class for labels
        with the following structure

        attributes:
            - category
            - score (True/False)
            - true score (int 1-10)
            - document
    """
    def __init__(self):
        self.cat = []
        self.score = []
        self.trueScore = []
        self.doc = []
        self.index = 0
    
    # the two following functions are
    # needed for making this object iterable
    def __iter__(self):
        return self

    def next(self):
        if self.index == len(self.cat)-1:
            raise StopIteration
        self.index = self.index + 1
        return (self.cat[self.index], self.score[self.index],
                self.trueScore[self.index], self.doc[self.index])

    def removeItems(self, elements):
        # removes items by a indexes
        multi_delete(self.cat, elements)
        multi_delete(self.score, elements)
        multi_delete(self.trueScore, elements)
        multi_delete(self.doc, elements)

class TextData:
    """
        parse json data with the following hirearchy:
            doc1
              category1
                [score]
                [sentences]
                  sent1
                  ..
                  sentN 
              ..
              categoryN
            ..
            docN
        
        Params:     jsonFname - data file name in json format
                    balance -   if True perform data 
                                balancing
                    withFirst - if False removes first 
                                setence (description)
                                from data

        Attributes  X - paragraph
                    Y - Feature (class repres' above)

    """
    def __init__(self,jsonFname,balance=False,withFirst=False):
        # parse data
        self.data = self.loadJson(jsonFname)
        # flag - take first sentence or not
        self.withFirst = withFirst
        # get x,y (n_samples, n_labels)
        self.x, self.y = self.getXY(self.data)
        # num of documents
        self.docs = self.getNumOfDocs()
        # categories
        self.catScore = [False,True]
        self.catCat = self.getCats(self.data)
        # balance x,y
        if balance:
            self.x, self.y = self.balancing(self.x, self.y) 

    def loadJson(self,fname):
        # load json file from current path
        with open(fname, 'rb') as f:
            # assumes one row in json file
            for row in f:
                json_data = row
        return json.loads(json_data)

    def getXY(self,textObj):
        # return X - paragraphs, Y - (category, paragraph, document)
        cats = self.getCats(textObj)
        X = []
        Y = Feature()
        # iterate over all docs
        for i in range(0, len(textObj)):
            # iterate over all categories
            for c in cats:
                # score
                score = self.binScore(textObj[i][c]['score'])
                trueScore = textObj[i][c]['score']
                cat = c
                label = (cat,score,i,trueScore)
                # iterate over all sentences
                paragraph = ''
                # continue if only one sentence and <= 5
                numSent = len(textObj[i][c]['sents'])
                if numSent:
                    firstSentLen = len(textObj[i][c]['sents'][0])
                if not self.withFirst and numSent == 1 and firstSentLen <= 5:
                    continue
                for k in range(0, numSent):
                    # continue if sentence is first and <= 5
                    if not self.withFirst and k==0 and firstSentLen <= 5:
                        continue    
                    # iterate over all words
                    # recreate sentence
                    numWords = len(textObj[i][c]['sents'][k])
                    for j in range(0, numWords):
                        # if not last word
                        if j != numWords-1:
                            paragraph += textObj[i][c]['sents'][k][j] + ' '
                        # last word
                        else:
                            paragraph += textObj[i][c]['sents'][k][j] + '\n'
                # append samples and labels
                X.append(paragraph)
                Y.cat.append(label[0])
                Y.score.append(label[1])
                Y.doc.append(label[2])
                Y.trueScore.append(label[3])
        return X,Y

    def getNumOfDocs(self):
        # return number of documets
        return self.y.doc[len(self.x)-1]+1

    def getCats(self,textObj):
        # return list of categories
        # (extracted from the first doc only)
        cats = []
        for i in range(0, len(textObj)):
            for key, value in textObj[i].iteritems():
                cats.append(key)
            break
        return cats

    def binScore(self,score):
        # binarize score
        # 1-5 -> Positive (True)
        # 6-10 -> Negative (False)
        if int(score) <= 5:
            return False
        elif int(score) > 5:
            return True
    # !!!
    # Some bug here !!!
    # !!!
    def balancing(self,x,y):
        # balance the dataset
        # by duplicating the smaller group (score)
        xres=[]
        yres=Feature()
        total = len(x)
        fCount, tCount = 0, 0
        for i in range(0,total):
            if y.score[i] == False:
                fCount += 1
        tCount = total-fCount

        if tCount < fCount:
            smaller = True
        elif tCount > fCount:
            smaller = False
        else:
            return x,y
        ftratio = float(fCount)/tCount
        # Positive is smaller
        length = len(x)
        if smaller:
            for i in range(0,length):
                if y.score[i] == True:
                    for s in range(0,int(ftratio)):
                        xres.append(x[i])
                        yres.cat.append(y.cat[i])
                        yres.score.append(y.score[i])
                        yres.doc.append(y.doc[i])
                        yres.trueScore.append(y.trueScore[i])
                else:
                    xres.append(x[i])
                    yres.cat.append(y.cat[i])
                    yres.score.append(y.score[i])
                    yres.doc.append(y.doc[i])
                    yres.trueScore.append(y.trueScore[i])
        # Negative  
        else:
            for i in range(0,length):
                if y.score[i] == False:
                    for s in range(0,int(1/ftratio)):
                        xres.append(x[i])
                        yres.cat.append(y.cat[i])
                        yres.score.append(y.score[i])
                        yres.doc.append(y.doc[i])
                        yres.trueScore.append(y.trueScore[i])
                else:
                    xres.append(x[i])
                    yres.cat.append(y.cat[i])
                    yres.score.append(y.score[i])
                    yres.doc.append(y.doc[i])
                    yres.trueScore.append(y.trueScore[i])

        return xres,yres

class Pipe:
    def __init__(self):
        self.svmpipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='modified_huber',penalty='l2',
                                  alpha=1e-3, n_iter=5,
                                  random_state=42)),
        ])

        self.nbpipeline = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),
        ])

def printRes(title,predicted,yTest,cats=None):
    # print results
    # accuracy, confusion, report
    print "______________________"
    print title
    print "______________________"
    acc = np.mean(predicted==yTest)
    print "Acc={0}".format(acc)
    print "Confusion Matrix:"
    print metrics.confusion_matrix(yTest,predicted)
    if cats != None:
        print "Classification Report:"
        print(metrics.classification_report(yTest, predicted,
                                            target_names=cats))
    print
    return acc

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