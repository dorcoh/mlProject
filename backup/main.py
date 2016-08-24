import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import numpy as np
import sys

def loadJson(fname):
	# load json file from current path
	with open(fname, 'rb') as f:
		# assumes one row in json file
		for row in f:
			json_data = row
	return json.loads(json_data)

def getCats(textObj):
	# return list of categories
	# (extracted from the first doc only)
	cats = []
	for i in range(0, len(textObj)):
		for key, value in textObj[i].iteritems():
			cats.append(key)
		break
	return cats

def getXY(textObj):
	# return X - paragraphs, Y - categories
	cats = getCats(textObj)
	X = []
	Y = []
	D = []
	# iterate over all docs
	for i in range(0, len(textObj)):
		# iterate over all categories
		for c in cats:
			# iterate over all sentences
			s = ''
			for item in textObj[i][c]['sents']:
				
				# iterate over all words
				# recreate sentence
				for j in range(0, len(item)):
					if j != len(item)-1:
						s += item[j] + ' '
					else:
						s += item[j] + '\n'
			D.append(i)
			X.append(s)
			Y.append(c)
	return X,Y

def getXYwithoutFirstSent(textObj):
	# return X - paragraphs, Y - categories
	cats = getCats(textObj)
	X = []
	Y = []
	# iterate over all docs
	for i in range(0, len(textObj)):
		# iterate over all categories
		for c in cats:
			# iterate over all sentences
			s = ''
			# continue if sentence is first and <= 3
			if len(textObj[i][c]['sents']) == 1 and len(textObj[i][c]['sents'][0]) <= 5:
				continue
			for k in range(0, len(textObj[i][c]['sents'])):
				# continue if sentence is first and <= 3
				if k==0 and len(textObj[i][c]['sents'][0]) <= 5:
					continue
				# iterate over all words
				# recreate sentence
				for j in range(0, len(textObj[i][c]['sents'][k])):
					if j != len(textObj[i][c]['sents'][k])-1:
						s += textObj[i][c]['sents'][k][j] + ' '
					else:
						s += textObj[i][c]['sents'][k][j] + '\n'
			X.append(s)
			Y.append(c)
	return X,Y

def binScore(score):
	# 1-5 -> Positive (True)
	# 6-10 -> Negative (False)
	if int(score) <= 5:
		return False
	elif int(score) > 5:
		return True

def getXYScore(textObj):
	# get X - paragraphs, Y - score
	cats = getCats(textObj)
	X = []
	Y = []
	# iterate over all docs
	for i in range(0, len(textObj)):
		# iterate over all categories
		for c in cats:
			Y.append(binScore(textObj[i][c]['score']))
			s = ''
			# iterate over all sentences
			for item in textObj[i][c]['sents']:	
				# iterate over all words
				# recreate sentence
				for j in range(0, len(item)):
					if j != len(item)-1:
						s += item[j] + ' '
					else:
						s += item[j] + '\n'
			X.append(s)
	return X,Y

def getXYScoreWithoutFirst(textObj):
	# get X - paragraphs, Y - score
	cats = getCats(textObj)
	X = []
	Y = []
	# iterate over all docs
	for i in range(0, len(textObj)):
		# iterate over all categories
		for c in cats:
			Y.append(binScore(textObj[i][c]['score']))
			s = ''
			# iterate over all sentences
			if len(textObj[i][c]['sents']) == 1 and len(textObj[i][c]['sents'][0]) <= 5:
				continue
			for k in range(0, len(textObj[i][c]['sents'])):
				# continue if sentence is first and <= 3
				if k==0 and len(textObj[i][c]['sents'][0]) <= 5:
					continue
				# iterate over all words
				# recreate sentence
				for j in range(0, len(textObj[i][c]['sents'][k])):
					if j != len(textObj[i][c]['sents'][k])-1:
						s += textObj[i][c]['sents'][k][j] + ' '
					else:
						s += textObj[i][c]['sents'][k][j] + '\n'
			X.append(s)
	return X,Y

def dupl(i, text):
	res=''
	for k in xrange(1,i):
		res = res + text
	return res

def balancingB(x,y):
	xres=[]
	yres=[]
	total = len(x)
	fCount, tCount = 0, 0
	for i in range(0,len(x)):
		if y[i] == False:
			fCount += 1
	tCount = total-fCount

	if tCount < fCount:
		smaller = True
	elif tCount > fCount:
		smaller = False
	else:
		return x,y

	# True is smaller
	length = len(x)
	if smaller:
		for i in range(0,length):
			if y[i] == True:
				xres.append(x[i])
				yres.append(y[i])
			else:
				if tCount>0:
					xres.append(x[i])
					yres.append(y[i])
					tCount -= 1
	else:
		for i in range(0,length):
			if y[i] == False:
				xres.append(x[i])
				yres.append(y[i])
			else:
				if fCount>0:
					xres.append(x[i])
					yres.append(y[i])
					fCount -= 1
	return xres,yres

def balancing(x,y):
	xres=[]
	yres=[]
	total = len(x)
	fCount, tCount = 0, 0
	for i in range(0,len(x)):
		if y[i] == False:
			fCount += 1
	tCount = total-fCount

	if tCount < fCount:
		smaller = True
	elif tCount > fCount:
		smaller = False
	else:
		return x,y
	ftratio = float(fCount)/tCount
	# True is smaller
	length = len(x)
	if smaller:
		for i in range(0,length):
			if y[i] == True:
				for s in range(0,int(ftratio)):
					xres.append(x[i])
					yres.append(y[i])
			else:
				xres.append(x[i])
				yres.append(y[i])
					
	else:
		for i in range(0,length):
			if y[i] == False:
				for s in range(0,int(1/ftratio)):
					xres.append(x[i])
					yres.append(y[i])
			else:
				xres.append(x[i])
				yres.append(y[i])

	return xres,yres

if __name__ == '__main__':
	# load data
	textObjTrain = loadJson('train-reviews.json')
	textObjTest = loadJson('test-reviews.json')

	xTrain, yTrain = getXYwithoutFirstSent(textObjTrain)
	xTest, yTest = getXYwithoutFirstSent(textObjTest)

	# create NB classifier pipeline
	text_clf_nb = Pipeline([('vect', CountVectorizer()),
						 ('tfidf', TfidfTransformer()),
						 ('clf', MultinomialNB()),
	])

	# create SVM classifier pipeline
	text_clf_svm = Pipeline([('vect', CountVectorizer()),
						 ('tfidf', TfidfTransformer()),
						 ('clf', SGDClassifier(loss='hinge', penalty='l2',
						  					   alpha=1e-3, n_iter=5,
						  					   random_state=42)),
	])
	
	print "_______________________________________________________"
	print "		predict aspects"
	print "_______________________________________________________"
	# naive bayes clf
	text_clf = text_clf_nb.fit(xTrain, yTrain)
	predicted = text_clf.predict(xTest)
	prob = text_clf.predict_proba(xTest)

	"""
	for p in prob:
		print text_clf.classes_, p
	"""

	print "Accuracy NB={0}".format(np.mean(predicted==yTest))
	print "Confusion matrix:"
	print metrics.confusion_matrix(yTest, predicted)

	# svm clf
	text_clf = text_clf_svm.fit(xTrain, yTrain)
	""" # - tuning
	parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
				  'tfidf__use_idf': (True, False),
				  'clf__alpha': (1e-2, 1e-3),
	}
	gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
	gs_clf = gs_clf.fit(xTrain, yTrain)
	predicted = gs_clf.predict(xTest)
	"""
	predicted = text_clf.predict(xTest)

	print "Accuracy SVM={0}".format(np.mean(predicted==yTest))
	print "Confusion matrix:"
	print metrics.confusion_matrix(yTest, predicted)



	print "_______________________________________________________"
	print "		predict scores"
	print "_______________________________________________________"

	xTrainScore, yTrainScore = getXYScoreWithoutFirst(textObjTrain)
	xTrainScore, yTrainScore = balancing(xTrainScore, yTrainScore)

	xTestScore, yTestScore = getXYScoreWithoutFirst(textObjTest)
	
	# naive bayes clf
	text_clf = text_clf_nb.fit(xTrainScore, yTrainScore)
	predicted = text_clf.predict(xTestScore)
	print "Accuracy NB={0}".format(np.mean(predicted==yTestScore))
	print "Confusion matrix:"
	print metrics.confusion_matrix(yTestScore, predicted)

	# svm clf
	text_clf = text_clf_svm.fit(xTrainScore, yTrainScore)
	"""
	parameters = [
		{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
		{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
	]
	gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
	gs_clf = gs_clf.fit(xTrain, yTrain)
	predicted = gs_clf.predict(xTest)
	"""
	predicted = text_clf.predict(xTestScore)
	print "Accuracy SVM={0}".format(np.mean(predicted==yTestScore))
	print "Confusion matrix:"
	print metrics.confusion_matrix(yTestScore, predicted)
	
	# Second pipeline

	print "_______________________________________________________"
	print "		predict prob aspect"
	print "_______________________________________________________"

	# create NB classifier pipeline
	text_clf_nb = Pipeline([('vect', CountVectorizer()),
						 ('tfidf', TfidfTransformer()),
						 ('clf', MultinomialNB()),
	])

	# create SVM classifier pipeline
	text_clf_svm = Pipeline([('vect', CountVectorizer()),
						 ('tfidf', TfidfTransformer()),
						 ('clf', SGDClassifier(loss='hinge', penalty='l2',
						  					   alpha=1e-3, n_iter=5,
						  					   random_state=42)),
	])

	text_clf = text_clf_svm.fit(xTrain, yTrain)

	predicted = text_clf.predict(xTestScore)
	predictedTraining = text_clf.predict(xTrainScore)
	"""
	prob = text_clf.predict_proba(xTest)
	probTrain = text_clf.predict_proba(xTrain)
	"""

	# create modified test and training set
	xTestMod = []
	for i in range(0, len(xTestScore)):
		xTestMod.append(dupl(1,"cat:" + predicted[i]+" ")+xTestScore[i])
	xTrainMod = []
	for i in range(0, len(xTrainScore)):
		xTrainMod.append(dupl(1," cat:" + predictedTraining[i])+xTrainScore[i])

	text_clf = text_clf_nb.fit(xTrainMod, yTrainScore)
	predicted = text_clf.predict(xTestMod)
	
	print "Accuracy NB={0}".format(np.mean(predicted==yTestScore))
	print "Confusion matrix:"
	print metrics.confusion_matrix(yTestScore, predicted)

	text_clf = text_clf_svm.fit(xTrainMod, yTrainScore)
	predicted = text_clf.predict(xTestMod)
	
	print "Accuracy SVM={0}".format(np.mean(predicted==yTestScore))
	print "Confusion matrix:"
	print metrics.confusion_matrix(yTestScore, predicted)

