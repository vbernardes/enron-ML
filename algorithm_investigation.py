#!/usr/bin/python

import sys
import pickle
sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, recall_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV

import numpy as np

def init_alg_investigation():
    # Initializing dataset for local testing
    features_list = ['poi',
                     'total_stock_value',
                     'total_payments',
                     'restricted_stock',
                     'exercised_stock_options',
                     'salary',
                     'expenses',
                     'other',
                     'to_messages',
                     'shared_receipt_with_poi',
                     'from_messages',
                     'from_this_person_to_poi',
                     'from_poi_to_this_person',
                     'bonus']

    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    data_dict.pop('TOTAL')

    my_dataset = data_dict

    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)


def test_classifier(clf, data_sets, features_list):
    iter_num = 0

    precision_list = []
    recall_list = []
    chosen_features = {}

    proportion_from_poi_scores = []

    for data_set in data_sets:
        features_train, labels_train, features_test, labels_test = data_set

        # Select features
        k = 4   # number of features
        selector = SelectKBest(f_classif, k=k)
        selector.fit(features_train, labels_train)

        # Show k top features:
        # print('Features Scores: '+str(sorted(zip(selector.scores_, features_list[1:]), reverse=True)[:k]))

        # Score for feature we created, for comparison purposes:
        for feat_score in zip(selector.scores_, features_list[1:]):
            if feat_score[1] == 'proportion_from_poi':
                # print('proportion_from_poi Score: '+str(feat_score[0]))
                proportion_from_poi_scores.append(feat_score[0])

        # Let's count how many times each feature comes up in the selector
        for selected_feat in sorted(zip(selector.scores_, features_list[1:]), reverse=True)[:k]:
            if selected_feat[1] in chosen_features.keys():
                chosen_features[selected_feat[1]] += 1
            else:
                chosen_features[selected_feat[1]] = 1

        selected_features_train = selector.transform(features_train)
        selected_features_test = selector.transform(features_test)

        # Fit classifier and make predictions
        clf.fit(selected_features_train, labels_train)
        pred_selected = clf.predict(selected_features_test)

        # Test for errors:
        precision = precision_score(labels_test, pred_selected)
        recall = recall_score(labels_test, pred_selected)
        precision_list.append(precision)
        recall_list.append(recall)

        # print('Iteration: '+str(iter_num))
        # print('Precision: '+str(precision))
        # print('Recall: '+str(recall))
        # print

        iter_num += 1

    # Print summary
    print('Classifier: '+str(clf))
    print('Average precison: '+str(np.mean(precision_list)))
    print('Average recall: '+str(np.mean(recall_list)))
    print('Average proportion_from_poi score: '+str(np.mean(proportion_from_poi_scores)))
    print
    print(sorted(chosen_features.items(), key=lambda x: x[1], reverse=True))
    print


def evaluate_classifier(clf, data_sets):

    precision_list = []
    recall_list = []

    for data_set in data_sets:
        features_train, labels_train, features_test, labels_test = data_set

        # Fit classifier and make predictions
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)

        # Test for errors:
        precision = precision_score(labels_test, predictions)
        recall = recall_score(labels_test, predictions)
        precision_list.append(precision)
        recall_list.append(recall)

    # Print summary
    print('Classifier: '+str(clf))
    print('Average precison: '+str(np.mean(precision_list)))
    print('Average recall: '+str(np.mean(recall_list)))
    print


def split_data_sss(features, labels, n_iter):
    '''Uses StratifiedShuffleSplit to create train and test data.
    Returns a list of tuples.'''
    sss = StratifiedShuffleSplit(y=labels, test_size=0.3, n_iter=n_iter)

    train_test_sets = []

    for train_index, test_index in sss:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []

        for i in train_index:
            features_train.append(features[i])
            labels_train.append(labels[i])
        for j in test_index:
            features_test.append(features[j])
            labels_test.append(labels[j])

        train_test_sets.append((features_train, labels_train, features_test, labels_test))

    return train_test_sets


def investigate_algorithms(features, labels, features_list):
    # Prepare Decision Tree to tune parameters
    DT_params = {
        'criterion':['gini', 'entropy'],
        'splitter':['best', 'random'],
        'max_depth':[1, 10, 100],
        'min_samples_split':[2, 10, 100]
    }
    DT = DecisionTreeClassifier(random_state=42)
    clf_DT = GridSearchCV(DT, DT_params)

    # Split data into train/test data sets
    data_sets = split_data_sss(features, labels, 100)

    # Test our classifiers
    classifiers = [
        GaussianNB(),
        clf_DT
    ]
    for clf in classifiers:
        test_classifier(clf, data_sets, features_list)

    # See DT parameters
    print('Chosen DT parameters: '+str(clf_DT.best_params_))