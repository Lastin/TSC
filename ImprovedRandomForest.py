import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
import math
import Queue
import thread
import pylab
import sys

#=============================================================================================================================

#TRAIN DATA
train_df = pd.read_csv('train.csv', header=0)

#Gender
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

#Embarked - N/A to most common
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))
Ports_dict = { name : i for i, name in Ports }
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

#Fare - grouping
train_df['Fare'] = train_df['Fare'].map(lambda x: 0 if x > 250 else (1 if x > 70 else (2 if x > 25 else 3)))

#250+ parlor suite (estimate)
#70+ 1st class
#25+ 2nd class
#all below is 3rd
#train_df['Class'] = train_df['Fare'].map(lambda x: 0 if x > 250 else (1 if x > 70 else (2 if x > 25 else 3)))

#Age - N/A to median of class and gender combination
median_ages = np.zeros((2,4))
for gn in xrange(0,2): #gender
    for cl in xrange(0,4): #class
        median_ages[gn, cl] = train_df[(train_df['Gender'] == gn) & (train_df['Pclass'] == cl)]['Age'].dropna().median()

for gn in xrange(0,2):
    for cl in xrange(0,4):
        train_df.loc[ (train_df.Age.isnull()) & (train_df.Gender == gn) & (train_df.Pclass == cl), 'Age'] = median_ages[gn, cl]

train_df['Age'] = train_df['Age'].map(lambda x: 0 if x < 5 else (1 if x < 10 else (2 if x < 16 else (3 if x < 40 else 4))))

#Drop not needed columns
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

#=============================================================================================================================

#TEST DATA
test_df = pd.read_csv('test.csv', header=0)
#Gender
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

#Embarked - N/A to most common
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
#string to int. Ref: line 17
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

#Fare - N/A to median of respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fares = np.zeros(4)
    for cl in range(0,3):                                              # loop 0 to 2
        median_fares[cl] = test_df[test_df.Pclass == cl]['Fare'].dropna().median()
    for cl in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == cl ), 'Fare'] = median_fares[cl]

test_df['Fare'] = test_df['Fare'].map(lambda x: 0 if x > 250 else (1 if x > 70 else (2 if x > 25 else 3)))

#Age - N/A to median
median_ages = np.zeros((2,4))
for gn in xrange(0,2):
    for cl in xrange(0,4):
        median_ages[gn, cl] = test_df[(test_df['Gender'] == gn) & (test_df['Pclass'] == cl)]['Age'].dropna().median()

for gn in xrange(0,2): #gender
    for cl in xrange(0,4): #class
        test_df.loc[ (test_df.Age.isnull()) & (test_df.Gender == gn) & (test_df.Pclass == cl), 'Age'] = median_ages[gn, cl]

test_df['Age'] = test_df['Age'].map(lambda x: 0 if x < 5 else (1 if x < 10 else (2 if x < 16 else (3 if x < 40 else 4))))
#=============================================================================================================================

#Collect test data before dropping it
ids = test_df['PassengerId'].values
names = test_df['Name'].values
#Drop columns
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

#Convert to a numpy array
train_data = train_df.values
test_data = test_df.values




def train(estimators, features):
    #print 'Training!'
    forest = RandomForestClassifier(n_estimators = estimators, max_features = int(math.sqrt(features)))
    forest = forest.fit(train_data[0::,1::], train_data[0::,0])
    return forest

def predict(forest):
    #print 'Predicting!'
    return forest.predict(test_data).astype(int)

def save(output):
    print 'Saving...'
    predictions_file = open("output.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()
    print 'Saving complete!'

output = predict(train(2000, 8))
save(output)
