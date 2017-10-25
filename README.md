# LangClassifiers-P2

All of the code was implemented on Python 3.6 and Sklearn 0.19.

classifiers.py implements most of the classification methods, namely:

'NB'      our own implementation of Naive Bayes
'RF'      sklearn's Random Forests Classifier
'AB-RF'   sklearn's Random Forests Classifier with Adaboost
'AB-SGD'  sklearn's Stochastic Gradient Descent Classifier with Adaboost
'NBM'     sklearn's Multinomial Naive Bayes
'MLP'     sklearn's Multi-layer Perceptron

The classifier type can be selected by changing the line

CLF_type = 'NB'

to match any of the options listed above.

Before running the classifier, you must first compute the feature dictionary using

    python featureExtractor.py

After featureExtractor was run, you must run

    python classifiers.py

Further information on the functionality of the individual components of the code can be found in each of the respective .py files as comments.

The output .csv file will be generated in the same folder as classifiers.py and will have a name in form of {CLASSIFIER_NAME}.csv, where 
CLASSIFIER_NAME will be a string describing the classifier type and some of its key hyperparameters.

##############################################################################################################################################

Code below runs on Python 2.

./Naive_Bayes_Bonnie 
    -->  Please check ./Naive_Bayes_Bonnie/README for running instructions 

##############################################################################################################################################

K-Nearest Neighbors

All of the code was implemented in Python 3.

1. Run featureExtractor.py with the desired datasets
    optional:
    - names can be specified in trainSetXFilename, trainSetYFilename, testSetXFilename
    - forceBuildNewDictionary and forceBuildNewBestDictionary are toggles for creating a new processed dataset or loading a previously made one
    - run the file to make sure the dictionary with the respective name is present in the data folder

2. Run kNN_kmeans:
    optional:
    - adjust parameters (k, num_clust)
    - include the name of the dictionary for vectorization as pickle_name
    - sort_train and sort_test - toggles for vectorizing new datasets or loading previously processed ones
    - raw_train is processing of the train set without the language sorting - required for validation check
    - as a default, the script will generate the test_y file for a given test_x. Set validation to True if you want to run a validation check in case your test set

The output file will be generated ("submission.csv").

3. Validation:
As a default, the script will generate the test_y file for a given test_x. If you require a validation check: see commented lines

