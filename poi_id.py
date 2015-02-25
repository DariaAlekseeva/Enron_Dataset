#!/usr/bin/python

import matplotlib.pyplot as plt
import sys
import pickle
from sklearn import preprocessing
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
sys.path.append("../tools/")


from feature_format import featureFormat
from feature_format import targetFeatureSplit

# function for calculation ratio of true positives
# out of all positives (true + false)
def precision(pred,labels_test):
    precision, precision_d, precision_n=0.0,0.0,0.0
    for i in range(0,len(pred)):
        if pred[i]==1:
            precision_d+=1
            if labels_test[i]==1:
                precision_n+=1
    if precision_d!=0:
        precision=precision_n/precision_d
    return precision

# function for calculation ratio of true positives
# out of true positives and false negatives
def recall(pred,labels_test):
    recall, recall_d, recall_n=0.0,0.0,0.0
    for i in range(0,len(pred)):
        if labels_test[i]==1:
            recall_d+=1
            if pred[i]==1:
                recall_n+=1
    if recall_d!=0:
        recall=recall_n/recall_d
    return recall

### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label
features_list = ["poi", "salary", "bonus", "fraction_from_poi_email", "fraction_to_poi_email", 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value']

### load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### look at data
#print len(data_dict.keys())
#print data_dict['BUY RICHARD B']
#print data_dict.values()


### remove any outliers before proceeding further
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

### remove NAN's from dataset
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
### uncomment for printing top 4 salaries
### print outliers_final


### plot features
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
#plt.show()




### create new features
### new features are: fraction_to_poi_email,fraction_from_poi_email

def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

### create two lists of new features
fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]=fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[count]
    count +=1


### store to my_dataset for easy export below
my_dataset = data_dict


### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)

### plot new features
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter( from_poi, to_poi )
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("fraction of emails this person gets from poi")
#plt.show()


### if you are creating new features, could also do that here


### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)




### machine learning goes here!
### please name your classifier clf for easy export below

### deploying feature selection
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

### use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

from sklearn.tree import DecisionTreeClassifier

t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
print 'accuracy before tuning ', score

print "Decision tree algorithm time:", round(time()-t0, 3), "s"



importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
#print 'Feature Ranking: '
#for i in range(16):
#    print "{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]])



### try Naive Bayes for prediction
#t0 = time()

#clf = GaussianNB()
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#accuracy = accuracy_score(pred,labels_test)
#print accuracy

#print "NB algorithm time:", round(time()-t0, 3), "s"


### use gridCV for tuning parameters
print("Searching the best parameters for Decision Tree")
t0 = time()
param_grid = {'criterion': ('gini','entropy'),
              'splitter':('best','random'),
              'min_samples_split':[2,3,4,5,6,8],
                'max_features':('auto','sqrt','log2',None),
                'max_depth':[None,1,2,10,50],
                'max_leaf_nodes':[None,2,5,6,7,8,9,10,13]}
clf = GridSearchCV(DecisionTreeClassifier(random_state=42),param_grid)
clf = clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
print("done in %0.3fs" % (time() - t0))
print("Best parameters found by grid search:")
print(clf.best_estimator_)

acc=accuracy_score(labels_test, pred)

print "Validating algorithm:"
print "accuracy after tuning = ", acc
print 'precision = ', precision(pred, labels_test)
print 'recall = ', recall(pred, labels_test)



### dump your classifier, dataset and features_list so
### anyone can run/check your results
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )