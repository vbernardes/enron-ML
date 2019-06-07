######
# INIT
######

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


##################
# DATA EXPLORATION
##################

# Let's take a look at our data set and get familiar with our data points
num_data_points = len(data_dict)

num_poi, num_no_poi = 0, 0
for person in data_dict.keys():
    if data_dict[person]['poi'] == 0:
        num_no_poi += 1
    else:
        num_poi += 1

print("Number of data points: " + str(num_data_points))
print("Number of POI: " + str(num_poi))
print("Number of non-POI: " + str(num_no_poi))
print("Proportion of POI to non-POI: " + str(num_poi/float(num_no_poi)))

# Count total features available "out of the box"
features_list = data_dict[data_dict.keys()[0]].keys()
print("Number of features available at the start: "
      + str(len(features_list) - 1))    # minus one to exclude 'poi' label
                                        # which is not a feature

# Are there features with many missing values?
missing_values = []

for feature in [f for f in features_list if f != 'poi']:
    num_nan = 0
    for person in data_dict.keys():
        feature_value = data_dict[person][feature]
        if feature_value == 'NaN':  # Missing values are encoded as 'NaN'
            num_nan += 1
    print("Feature '"+feature+"' has "+str(num_nan)+" missing values out of "
          +str(num_data_points)+", or "+str(num_nan/float(num_data_points))+"")
    missing_values.append((feature, num_nan, num_data_points, num_nan/float(num_data_points)))


# Prep report table
print("| Feature | # Missing Values | Proportion of Missing Values |") # header
print("|:--|:--|:--|") # formatting
# Sort list of features according to number of missing values:
for row in sorted(missing_values, key=lambda x: x[3]):
    print("| "+str(row[0])+" | "+str(row[1])+" | "+str(round(row[3], 3))+" |")


# Investigate amount of usable data for each data point
TOTAL_FEATURES = len(data_dict[data_dict.keys()[0]].keys()) - 1 # minus 1 to exclude POI indicator
proportion_unusable_data = {}
for person, data in data_dict.items():
    num_NaN = 0
    for key, value in data.items():
        if key != 'poi':
            if value == 'NaN':
                num_NaN += 1
    proportion_unusable_data[person] = num_NaN / float(TOTAL_FEATURES)

# Let's see how much unusable data each person has
print(sorted(proportion_unusable_data.items(), key=lambda x: x[1], reverse=True))


##########
# OUTLIERS
##########

def load_features(features_list):
    "Auxiliary function to load new features from data set."
    ### The first feature in features_list must be "poi".
    ### Extract features and labels from dataset for local testing
    data = featureFormat(data_dict, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    return labels, features

def plot_features(features, labels, features_list):
    ### The first feature in features_list must be "poi".

    # Color POIs as red and non-POIs as blue
    color_map = map(lambda x: 'b' if x == 0 else 'r', labels)

    plt.scatter([f[0] for f in features],
               [f[1] for f in features],
               c=color_map)
    plt.xlabel(features_list[1])
    plt.ylabel(features_list[2])


def get_n_max_val(features, features_list, n=1):
    """Return n maximum values for features.
    The first feature in features_list must be "poi"."""

    max_val = []
    for i in range(len(features_list[1:])):
        values = [feat[i] for feat in features]
        max_val.append((features_list[i+1], sorted(values)[-n:]))

    return max_val


list_of_features_lists = [
    ['poi','total_stock_value', 'total_payments'],
    ['poi', 'restricted_stock', 'exercised_stock_options'],
    ['poi', 'salary', 'expenses']
]

plt.figure(figsize=(6,7))

i = 1  # subplot index
for features_list in list_of_features_lists:
    labels, features = load_features(features_list)

    plt.subplot(3,1,i)
    plot_features(features, labels, features_list)

    i += 1

    # Print max value for current features
    print(get_n_max_val(features, features_list))

plt.tight_layout()
plt.savefig('exploration_outliers.png', dpi=400)
