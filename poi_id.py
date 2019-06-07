#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
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

### Task 2: Remove outliers
# Remove outliers from our data set
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('LOCKHART EUGENE E')


### Task 3: Create new feature(s)
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

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Trying two classifiers:
from algorithm_investigation import investigate_algorithms
investigate_algorithms(features, labels, features_list)

# After testing, the one with the best results is implemented below:
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)
from algorithm_investigation import split_data_sss, evaluate_classifier

# Split data into train/test data sets
data_sets = split_data_sss(features, labels, 1000)

evaluate_classifier(clf, data_sets)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)