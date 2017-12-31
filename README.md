A machine learning study about the UCI's abalone dataset

http://archive.ics.uci.edu/ml/datasets/Abalone?pagewanted=all

To install the python requirements:

sudo pip install -r requirements.txt

File description:

data - Folder with the original UCI's files

docs - Folder with the project's tasks (portuguese)

result_images - Folder with the images generated throughout the study

result_pickles - Folder with the pickle files generated throughout the study

CombinedClassifier.py - Combined classifiers that learns in other's classifiers output space

compare_all_classifiers.py - Compares the accuracy of all classifiers in a paired test

compare_all_classifiers_plus_combined.py - Compare the mlp classifier with the combined one

compare_all_classifiers_time.py - Compare all classifiers regarding the processing time

covariance_all_classifiers.py - Computes the covariance between the classifiers output

fatures.pickle - Saved random feature mapping

investigate_data.py - Create scatter plots with the initial data

MajorityVoteClassifier.py - Majority vote classifier

make_friedman_and_nemenyi_test.py - Does the hypothesis testing

MLP.py - Custom neural network

random_tansformations.py - Creates random feature mappings

test_knn_dist_weight.py - Test the KNN ponderation possibilities

test_knn_num_bins.py - Test the number of bins to be used with a 7-neighbor Knn

test_knn_y_conv_accuracy.py - Test the number of neighbors to use in KNN with accuracy as score

test_knn_y_conv_f1_score.py - Test the number of neighbors to use in KNN with f1-measure as score

test_knn_y_conv_inbalance.py - Tests wilson editing

test_mlp_activation.py - Tests different activation functions to use with MLP

test_mlp_learning_rates.py - Tests different learning rates with MLP

test_mlp_max_iter.py - Tests different max iterations with MLP

test_mlp_num_layers.py - Tests the number of hidden layers to be used with MLP

test_mlp_regularization.py - Test the MLP's regularization term

test_mlp_y_conv_f1_score.py - Tests MLP's hidden layer size by using f1-measure as score

test_self_mlp.py - Tests the custom MLP

test_svm_y_conv_f1_score.py - Tests the C-value on a SVM classifier

test_svm_kernel.py - Tests the Kernel for a SVM classifier

test_tree_max_depth.py - Tests the decision tree max depth

test_tree_max_leaf.py - Tests the max number of leafs in a decision tree classifier

test_tree_metric.py - Tests the split metric options for a decision tree classifier

util.py - Util functions used by many files
