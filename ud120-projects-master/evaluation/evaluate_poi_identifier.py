"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
from sklearn import tree
import sys
sys.path.append("/Users/alitabet/Documents/Online Course/IDrive-Sync/Udacity - Intro to Machine Learning/ud120-projects-master/tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from sklearn import cross_validation
from sklearn import metrics

data_dict = pickle.load(open("final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
print "Accuracy with CV is",clf.score(features_test,labels_test)

pred = clf.predict(features_test,labels_test)
print "Number of identified POIs =",pred.sum()

print "Number of people in test set is",len(labels_test)

print "Accuracy of all zero prediction =",1.0*(len(labels_test) - pred.sum())/len(labels_test)

print labels_test
print pred

print "Precision =",metrics.precision_score(labels_test,pred)
print "Recall =",metrics.recall_score(labels_test,pred)
