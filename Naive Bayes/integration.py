import featureExtractor as fe
import naive_bayes as nb
import time



# run ngram generator for both training, validation, and test set
'''
Remeber! set NUM_SAMPLES, MIN_GRAM, MAX_GRAM parameters in featureExtractor.py
'''
data = fe.generate_freq()
test = fe.getTestData()

# run traing and prediction in Naive Bayes
'''
Remeber! set VALIDATION_FRAC, TOP_N_FEATURE, TRIAL in naive_bayes.py
         set NUM_SAMPLES, MIN_GRAM, MAX_GRAM the same as featureGenerator.py
'''
t0 = time.time()
nb.run_prediction(data, test)
print 'running time: ', time.time()-t0



