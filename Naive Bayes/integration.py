import featureExtractor as fe
import naive_bayes as nb
import time

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



