import time
import featureExtractor as fe
import naive_bayes as nb


dataFolderName = "data/"

# Set training files
trainSetXFilename = "generatedTestSetX-200000.csv"
trainSetYFilename = "generatedTestSetY-200000.csv"
#trainSetXFilename = "train_set_x_clean.csv"
#trainSetYFilename = "train_set_y_clean.csv"

# Set test file
testFilename = 'test_set_x.csv'

# Set hyper parameters
# how many samples to use for training
NUM_SAMPLES = 200000
# Set N-gram: lower and upper bound
MIN_GRAM = 1
MAX_GRAM = 1
# Set validation set fraction of training set
VALIDATION_FRAC = 0.05
# Set number of top frequency features to keep in training
TOP_N_FEATURE = 100
# Set test Trial for test output generating
TRIAL = 'trail_0'

# if you want to skip training and loading existing model, please set load_model=True, point modelFile to the one you want
load_model = True
modelFile = 'model_190000Train_113Feature.pkl'

# if test result available, simply give its path to testResultFile
testResultFile = None


# generate n-gram features for both training and testing sets
train = fe.generate_freq(NUM_SAMPLES, dataFolderName,trainSetXFilename, trainSetYFilename, MIN_GRAM, MAX_GRAM)
test = fe.getTestData(dataFolderName,testFilename,MIN_GRAM, MAX_GRAM)

# run traing and prediction in Naive Bayes
t0 = time.time()
nb.run_prediction(train, test,VALIDATION_FRAC, TOP_N_FEATURE, TRIAL, load_model=load_model, modelFile=modelFile, dataFolderName=dataFolderName,testResultFile=testResultFile)
print ('running time: ', time.time()-t0)



