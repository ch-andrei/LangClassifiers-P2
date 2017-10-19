import featureExtractor as fe
import naive_bayes as nb
import time
import pandas as pd
'''
Parameters to set before start:
- featureExtractor.py
    NUM_SAMPLES  -- samples used for traing and validation
    MIN_GRAM  --  Ngram lower limit
    MAX_GRAM  --  Ngram upper limit
- naive_bayes.py
    VALIDATION_FRAC  --  fraction of NUM_SAMPLES use for validation
    TOP_N_FEATURE  --  Suggestion: 75. Only use top N features (with high frequency) in each class to avoid overfitting
    TRIAL  --  Trial number to track test results
'''


# run ngram generator for both training, validation, and test set
data = fe.generate_freq()
test = fe.getTestData()


# run traing and prediction in Naive Bayes
t0 = time.time()
nb.run_prediction(data, test)
print ('running time: ', time.time()-t0)


# get accuracy
dft = pd.read_csv('C:\Users\guanqing\OneDrive\School\FALL 2017\COMP 551\Assignment2\LangClassifiers-P2\Naive Bayes\\test_result_10.csv')
dfa = pd.read_csv('C:\Users\guanqing\OneDrive\School\FALL 2017\COMP 551\Assignment2\LangClassifiers-P2\Naive Bayes\data\generatedTestSetY-100000.csv')

dfr = dft.merge(dfa, on='Id',how='left')
print dfr.head()
dfr['result'] = dfr.Category_x == dfr.Category_y
print float(len(dfr[dfr.result==True]))/len(dfr), len(dfr[dfr.result==True])

