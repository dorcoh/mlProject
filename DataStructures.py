"""
    Helper module for implementation of classes, functions
"""
import json
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
import numpy as np

catDict = {
    u'movie': 0,
    u'extras': 1,
    u'video': 2,
    u'audio': 3
}

def multi_delete(list_, elems):
    # delete multiple elements from a list
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

class Pipe:
    """
        Implements an SVM/naive bayes pipeline
        Using scikit-learn library
    """
    def __init__(self):
        self.svmpipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='log',penalty='l2',
                                  alpha=1e-3, n_iter=5,
                                  random_state=42)),
        ])

        self.nbpipeline = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),
        ])

def groupData(x,y,numOfDocs):
    # group scores by document, e.g [6,5,4,9]

    scores = [[0,0,0,0] for i in range(numOfDocs)]
    for i in range(0, len(x)):
        scores[y.doc[i]][catDict[y.cat[i]]] = y.score[i]

    return scores

def balancing(x,y):
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
    
    if fCount==0 or tCount==0 :
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