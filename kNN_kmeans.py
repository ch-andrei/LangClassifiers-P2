import numpy as np
import featureExtractor as fe
import pickle
import multiprocessing as mp
from multiprocessing import Process
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
#import pandas as pd
from collections import Counter
import queue

useAllCpus = True
if useAllCpus:
    numProcesses = mp.cpu_count()
else:
    numProcesses = max(1, int(mp.cpu_count() / 2)) # will use only half of the available CPUs

minLinesPerProcess = 1024 # if less than this, will only use 1 CPU
maxLinesBeforeDictIsEmptied = 4096 # will build dictionaries in increments of this # RAM saver

#Chosen parameters #######################################################################################################
k = 4 #number of nearest neighbors
num_clust = 10  #number of clusters for each language
pickle_name = "dict_1-3grams_276518Train.pkl"
sort_train = False
BEST_FEATURES_FORCE_RECOMPUTE = False
FULL_DICT_FORCE_RECOMPUTE = False

#Batching################################################################################################################

trainTotalSize = fe.getTrainXFileLineCount()
trainBatchSize = 1024 * 16
trainBatchCount = int(np.ceil(trainTotalSize / trainBatchSize))

testTotalSize = fe.getTestXFileLineCount()

trainValidationRatio = 0.15
trainValidationBatchCount = int(np.ceil(trainBatchCount * trainValidationRatio))
trainBatchCount -= trainValidationBatchCount

# predicting
predictTotalSize = fe.getTestXFileLineCount()
predictBatchSize = 1024 * 16
predictBatchCount = int(np.ceil(predictTotalSize / predictBatchSize))

print("Training with batch sample counts [train {} - validation {}], batch size {}".format(
    trainBatchCount * trainBatchSize, trainValidationBatchCount * trainBatchSize, trainBatchSize))

#Vectorize each class#####################################################################################################

ngram_dict = fe.pickleReadOrWriteObject(pickle_name, None)

ngramsList, ngramDict, ngramLangCounts = fe.processDictionaryAsTrainingFeatures(force_recompute_best=BEST_FEATURES_FORCE_RECOMPUTE, force_recompute_dict=FULL_DICT_FORCE_RECOMPUTE)

print("Vectorizing dictionary with {} best features".format(len(ngramsList)))
#for ngram in ngramsList:
    #print (ngram)  

def processXtrain ():
    tXs = queue.Queue()
    tYs = queue.Queue()
    batchNum = 0
    while batchNum < trainBatchCount:
        rawtX, tY = fe.readRawTrainingLines(trainTotalSize, trainBatchSize, batchNum * trainBatchSize)
        tX = fe.vectorizeLines(rawtX, ngramDict)
        tXs.put(tX)
        tYs.put(tY)
        batchNum += 1
    tX = tXs.get()
    tY = tYs.get()
    return sortXtrain(tX, tY)
    
def sortXtrain (rawtX, tY):
    sorted_lines = {0: [], 1: [], 2: [], 3: [], 4: []}
    for i in range(len(tY)):
            lang = tY[i]
            sorted_lines[lang].append(rawtX[i])
    return sorted_lines

def processXtest():
    testXs = queue.Queue()
    batchtestNum = 0
    while batchtestNum < trainBatchCount:
        testX = fe.readRawTestingLines(testTotalSize, trainBatchSize, batchtestNum * trainBatchSize)
        vectorized_test = fe.vectorizeLines(testX, ngramDict)
        testXs.put(vectorized_test)
        batchtestNum += 1
    testX = testXs.get()
    return testX
        
if sort_train == True:
    processedXtrain = processXtrain()
    pickle_out = open("processedXtrain.pkl","wb")
    pickle.dump(processedXtrain, pickle_out)
    pickle_out.close()
else:
    processedXtrain = pickle.load(open("processedXtrain.pkl", "rb"))
    print ('Loaded processed dictionary')
    
#kmeans-clustering#######################################################################################################       

def cluster_kmeans (x, n):
    kmeans = KMeans(n_clusters = n, random_state = 0).fit(x)
    return kmeans

#def cluster_centers (kmeans):
    #cluster_centers = kmeans.cluster_centers_
    #return cluster_centers  
   
#def cluster_labels (kmeans):
    #return kmeans.labels_
    
def cluster_lang (x):
    kmeans = cluster_kmeans (x, num_clust)   
    cluster_centers = kmeans.cluster_centers_
    print ('Computed clusters!')
    return cluster_centers


lang_0 = cluster_lang (processedXtrain[0])
lang_1 = cluster_lang (processedXtrain[1])
lang_2 = cluster_lang (processedXtrain[2])
lang_3 = cluster_lang (processedXtrain[3])       
lang_4 = cluster_lang (processedXtrain[4]) 

#combine everything into a training set

train_x = []
train_y = []
test_x = processedXtrain

def combine(x, y, tx, ty):
    lang = y
    lines = len(x) 
    for i in range(lines):
        tx.append(x[i])
    ty.extend([lang]*lines)
    return tx, ty

combine(lang_0, 0, train_x, train_y)
combine(lang_1, 1, train_x, train_y)
combine(lang_2, 2, train_x, train_y)       
combine(lang_3, 3, train_x, train_y)  
combine(lang_4, 4, train_x, train_y)        
  
#kNN#####################################################################################################################

predicted = []
#predicted.append("Id,Category")

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

def predict(train_x, train_y, test_x, k):
    distances = []
    labels = []
    for i in range(len(train_x)):
        dist= np.sqrt(np.sum(np.square(test_x - train_x[i])))
        distances.append([dist, i])
    distances = sorted(distances)
    for i in range(k):
        index = distances[i][1]
        labels.append(train_y[index])
    return Counter(labels).most_common(1)[0][0]
     
def kNN(train_x, train_y, test_x, predicted, k):
    for i in range(len(test_x)):
        predicted.append(predict(train_x, train_y, test_x[i], k))
    return predicted

""" 
batchNum = 0
while batchNum < trainBatchCount:
    test_x = processXtest ()
    test_y = kNN(train_x, train_y, val_x, predicted, k)
    for index, predict in enumerate(test_y):
        predictLine = "{},{}".format(batchNum * predictBatchSize + index, predict)
        predicted.append(predictLine)
        batchNum += 1
        print("\rPredicting progress {}/{}...".format(batchNum, predictBatchCount), end="")
    print('')
    with open("submission.csv", "w") as outputFile:
        for line in predicted:
            outputFile.write("{}\n".format(line))
"""
test_y = kNN(train_x, train_y, val_x, predicted, k)
acc_pred = np.asarray(predicted)

print (accuracy_score(val_y, acc_pred))
