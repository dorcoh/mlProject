# Machine Learning Project

##Goal 

To implement sentiment analysis classifier, with multiple aspects.

##Data
Dvd-reviews, multiple aspect for each review (Movie,Extras,Audio,Video) and scores for each one of the aspects (1-10)

##Process

First implemented as baseline three svm classifiers (aspect classifier, sentiment classifier - trained by score/trained by sentiment) using scikit-learn library

Another 'norma' classifier, which checks how far is some review in the test set from all other training reviews in terms of scores (1-10).

And finally three custom classifiers using probabilistic graphs, one directed model, and two undicreted.

##Results
In outputs directory
