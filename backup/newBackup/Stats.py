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

    histVar, binsVar = np.histogram(var, bins=40, range=(0,20), density=True)
    histDiff, binsDiff = np.histogram(diff, bins=40, range=(-10.0,10.0), density=True)

    return histVar,binsVar,histDiff,binsDiff

def plotStats(xTrain,yTrain,docsTrain,xTest,yTest,docsTest):
    # plot stats
    f, axarr = plt.subplots(2)

    # train data
    histVar, binsVar, histDiff, binsDiff = printStats(xTrain,yTrain,docsTrain)

    # subplot
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
    
    # plot
    plt.setp([a.get_xticklabels() for a in axarr[0:]], visible=False)
    plt.suptitle("Assumption checking: Paragraph scores are dependent on avg score (of it's document)")
    plt.show()