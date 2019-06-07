#!/usr/bin/python

import sys
import pickle
sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Select features to be used.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'total_stock_value',
                 'exercised_stock_options',
                 'salary',
                 'bonus']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# Remove outliers from our data set
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('LOCKHART EUGENE E')



# Creates a new feature: The proportion of all emails received by this person
# that were sent from a POI.
FEATURE_NAME = 'proportion_from_poi'
features_list.append(FEATURE_NAME)
for data in data_dict.values():
	if data['from_poi_to_this_person'] != 'NaN' and data['to_messages'] != 'NaN':
		data[FEATURE_NAME] = float(data['from_poi_to_this_person'])/float(data['to_messages'])
	else:
		data[FEATURE_NAME] = 'NaN'

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



# Trying two classifiers:
from algorithm_investigation import investigate_algorithms
investigate_algorithms(features, labels, features_list)

# After testing, the one with the best results is implemented below:
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Tune classifier
from algorithm_investigation import split_data_sss, evaluate_classifier

# Split data into train/test data sets
data_sets = split_data_sss(features, labels, 1000)

evaluate_classifier(clf, data_sets)

### Dump classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, features_list)