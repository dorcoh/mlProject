import json
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

class Feature:
    """
        a class for labels
        with the following structure

        attributes:
            - category
            - score
            - document
    """
    def __init__(self):
        self.cat = []
        self.score = []
        self.trueScore = []
        self.doc = []

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

        Attributes  X - paragraph,
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
            ('clf', SGDClassifier(loss='hinge',penalty='l2',
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
    print "Acc={0}".format(np.mean(predicted==yTest))
    print "Confusion Matrix:"
    print metrics.confusion_matrix(yTest,predicted)
    if cats != None:
        print "Classification Report:"
        print(metrics.classification_report(yTest, predicted,
                                            target_names=cats))
    print

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

    histVar, binsVar = np.histogram(var, bins=40, range=(0,20), density=True)
    histDiff, binsDiff = np.histogram(diff, bins=40, range=(-10.0,10.0), density=True)

    return histVar,binsVar,histDiff,binsDiff
    """
    zero = 0
    neg = 0
    pos = 0
    step = 0.5
    histArt = [0] * int(math.ceil(20/step))
    for j in range(0, len(x)):
        # diff between the true score and the average score
        # in current paragraph's document
        diff = y.trueScore[j] - sums[y.doc[j]]
        if diff > 0:
            pos += 1
        elif diff < 0:
            neg += 1
        else:
            zero += 1
        histArt[int(math.floor(diff+len(histArt)/2))] += 1

    # [-10, -9.5, ... , 9.5,10]
    bin_edges = []
    bin = -10
    for j in range(0, len(histArt)+1):
        bin_edges.append(bin+j*step)

    print bin_edges

    print "Negative: {0} ({1})\nZero: {2} ({3})\nPositive: {4} ({5})".format(
                                                    float(neg)/len(xTrain),neg,
                                                    float(zero)/len(xTrain),zero,
                                                    float(pos)/len(xTrain),pos)

    for i in range(0,len(histArt)):
        histArt[i] = float(histArt[i])/len(x)


    #hist, bin_edges = np.histogram(histArt, range=(-10.0,10.0), bins=40)
    #print hist, '\n', bin_edges

    plt.bar(bin_edges[:-1], histArt, width=step)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.show()    
    """

def plotStats(xTrain,yTrain,docsTrain,xTest,yTest,docsTest):

    f, axarr = plt.subplots(2)

    # train data
    histVar, binsVar, histDiff, binsDiff = printStats(xTrain,yTrain,docsTrain)

    axarr[0].bar(binsVar[:-1], histVar, width=0.5)
    axarr[0].set_title("Training - Variance")
    axarr[0].set_xlabel("Score's variance")
    axarr[0].set_ylabel("Probability")
    axarr[0].set_ylim([0, 0.5])
    axarr[1].bar(binsDiff[:-1], histDiff, width=0.5)
    axarr[1].set_title("Training - Difference")
    axarr[1].set_xlabel("Score's diff")
    axarr[1].set_ylabel("Probability")
    axarr[1].set_ylim([0, 0.5])
    
    """
    # test data
    histVar, binsVar, histDiff, binsDiff = printStats(xTest,yTest,docsTest)

    axarr[1, 0].bar(binsVar[:-1], histVar, width=0.5)
    axarr[1 ,0].set_title("Test - Variance")
    axarr[1, 0].set_xlabel("Score's variance")
    axarr[1, 0].set_ylabel("Probability")
    axarr[1, 0].set_ylim([0, 0.5])
    axarr[1, 1].bar(binsDiff[:-1], histDiff, width=0.5)
    axarr[1 ,1].set_title("Test - Difference")
    axarr[1, 1].set_xlabel("Score's diff")
    axarr[1, 1].set_ylabel("Probability")
    axarr[1, 1].set_ylim([0, 0.5])
    """
    
    # plot
    plt.setp([a.get_xticklabels() for a in axarr[0:]], visible=False)
    #plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    plt.suptitle("Assumption checking: Paragraph scores are dependent on avg score (of it's document)")
    plt.show()

    """
    plt.bar(bin_edges[:-1], histDiff, width=0.5)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.title("Probabilites of score variance for each document")
    plt.xlabel("Score's variance")
    plt.ylabel("Probability")
    plt.show()
    """

if __name__ == '__main__':
    trainData = TextData('train-reviews.json',balance=False)
    testData = TextData('test-reviews.json',balance=False)

    xTrain, yTrain, docsTrain = trainData.x, trainData.y, trainData.docs
    xTest, yTest, docsTest = testData.x, testData.y, testData.docs

    #printStats(xTrain,yTrain,docsTrain)
    #printStats(xTest,yTest,docsTest)
    #plotStats(xTrain,yTrain,docsTrain,xTest,yTest,docsTest)

    # category dictionary
    catDict = {
        u'movie': 0,
        u'extras': 1,
        u'video': 2,
        u'audio': 3
    }

    # append all document's scores to vector of size 4
    # [scoreMovie, scoreExtras, scoreVideo, scoreAudio]
    p = Pipe()
    clf = p.svmpipeline.fit(xTrain, yTrain.trueScore)
    predicted = clf.predict(xTest)

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
    #print newPredicted
    newPredicted = np.array(newPredicted)
    printRes("SVM-**SCORE**",newPredicted,yTest.score)
    #print predicted
    #print type(predicted), type(yTest.score)
    printRes("SVM-true score",predicted,yTest.score)


    """
    # tfidf transformer
    
    count_vect = CountVectorizer()
    xTrainCounts = count_vect.fit_transform(xTrain)
    print xTrainCounts.shape


    tf_transformer = TfidfTransformer(use_idf=False).fit(xTrainCounts)
    xTrainTf = tf_transformer.transform(xTrainCounts)
    """

    # predict
    """
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
    #printRes("SVM-true score",predicted,yTest.trueScore)
    """