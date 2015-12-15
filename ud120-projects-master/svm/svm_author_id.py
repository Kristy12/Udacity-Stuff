#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 


#########################################################
### your code goes here ###

from sklearn import svm

C_list = [10000.0]#[10.0,100.0,1000.0,10000.0]

for C_val in C_list:
    print "C = ",C_val
    clf = svm.SVC(C=C_val,kernel="rbf")
    to = time()
    clf.fit(features_train, labels_train)
    print "training time: ",round(time()-to,3),"s"
    t1 = time()
    pred = clf.predict(features_test)
    print "testing time: ",round(time()-t1,3),"s"
    from sklearn.metrics import accuracy_score
    print accuracy_score(pred,labels_test)
#########################################################


