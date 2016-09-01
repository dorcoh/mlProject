# Machine Learning Project

##Goal: 

to implement sentiment analysis classifier, with multiple aspects

Data: Dvd



We had a data of dvd-reviews, with multiple aspect for each review (Movie,Extras,Audio,Video) and scores for each one of the aspects, our goal was to implement sentiment analysis classifiers for each aspect, and an aspect classifier.

We first implemented as baseline three classifiers using scikit-learn library

Then we implemented a 'norma' classifier, which checks how far is some review in the test set from all other training reviews in terms of scores (1-10).

And finally we implemented three more classifiers assisted by the material we learned at the course (PGM)
where the difference between them are the kind of dependecies between the variables

Results so far:
```
______________________
SVM-category
______________________
Acc=0.992307692308
Confusion Matrix:
[[64  0  1  0]
 [ 0 64  1  0]
 [ 0  0 65  0]
 [ 0  0  0 65]]
 
Classification Report:
             precision    recall  f1-score   support

      movie       1.00      0.98      0.99        65
     extras       1.00      0.98      0.99        65
      video       0.97      1.00      0.98        65
      audio       1.00      1.00      1.00        65

avg / total       0.99      0.99      0.99       260


______________________
SVM-score (Trained on score)
______________________
Acc=0.838461538462
Confusion Matrix:
[[ 30  40]
 [  2 188]]

______________________
SVM-score (Trained on rating: 1-10)
______________________
Acc=0.788461538462
Confusion Matrix:
[[ 15  55]
 [  0 190]]

______________________
Norma CLF: SVM-SmoothedScore
______________________
Acc=0.803846153846
Confusion Matrix:
[[ 19  51]
 [  0 190]]

______________________
One-direction-dependency
______________________
Acc=0.807692307692
Confusion Matrix:
[[ 21  49]
 [  1 189]]
 
______________________
Pairwise-trained by all aspects
______________________
Acc=0.85
Confusion Matrix:
[[ 32  38]
 [  1 189]]
 
______________________
Pairwise-trained by aspect
______________________
Acc=0.846153846154
Confusion Matrix:
[[ 35  35]
 [  5 185]]

```
