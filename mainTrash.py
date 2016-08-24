if __name__ == '__main__':
    trainData = TextData('train-reviews.json',balance=False)
    testData = TextData('test-reviews.json',balance=False)

    xTrain, yTrain, docsTrain = trainData.x, trainData.y, trainData.docs
    xTest, yTest, docsTest = testData.x, testData.y, testData.docs
    
    #clfList, allSet = genClfs(xTrain,yTrain,xTest,yTest)
    #predSamp(xTest,yTest,23,clfList,allSet)
    #genClfsNew(xTrain,yTrain,xTest,yTest)
    sClassifier(xTrain,yTrain,xTest,yTest)
    superbClassifier(xTrain,yTrain,xTest,yTest,[0,1,2,3])
    #svmClassifier(xTrain, yTrain, docsTrain, xTest, yTest, docsTest)
    
    """
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
    """
    # assumption checking
    # plotStats(xTrain,yTrain,docsTrain,xTest,yTest,docsTest)
    
    # filtering
    
    #badIndexes = filterData(xTrain,yTrain,0,False)
    newX  = copy.deepcopy(xTrain)
    newY = copy.deepcopy(yTrain)
    #newY.removeItems(badIndexes)

    print len(newX), len(newY.cat)
    badIndexes = filterData(newX, newY, 1, False)
    multi_delete(newX,badIndexes)
    newY.removeItems(badIndexes)
    print len(newX), len(newY.cat)
    

    """
    badIndexes = filterData(newX, newY, 2, True)
    multi_delete(newX,badIndexes)
    newY.removeItems(badIndexes)
    print len(newX), len(newY.cat)
    """
    

    # classifiers
    #svmClassifier(xTrain, yTrain, docsTrain, xTest, yTest, docsTest)
    #normaClassifier(xTrain, yTrain, docsTrain, xTest, yTest, docsTest)