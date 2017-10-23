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
import csv
import operator
import pandas

useAllCpus = True
if useAllCpus:
    numProcesses = mp.cpu_count()
else:
    numProcesses = max(1, int(mp.cpu_count() / 2)) # will use only half of the available CPUs

minLinesPerProcess = 1024 # if less than this, will only use 1 CPU
maxLinesBeforeDictIsEmptied = 4096 # will build dictionaries in increments of this # RAM saver

#Chosen parameters #######################################################################################################
k = 4 #number of nearest neighbors
num_clust = 50 #number of clusters for each language
raw_train = False
pickle_name = "ngramListDictCounts_max10000_1-1.pkl" #ngram dictionary name
sort_train = True #process and sort the training set?
sort_test = True 
BEST_FEATURES_FORCE_RECOMPUTE = False
FULL_DICT_FORCE_RECOMPUTE = False #process and sort the test set?


#Vectorize each class#####################################################################################################

ngram_dict = fe.pickleReadOrWriteObject(pickle_name, None)

ngramsList, ngramDict, ngramLangCounts = fe.processDictionaryAsTrainingFeatures(force_recompute_best=BEST_FEATURES_FORCE_RECOMPUTE, force_recompute_dict=FULL_DICT_FORCE_RECOMPUTE)

print("Vectorizing dictionary with {} best features".format(len(ngramsList)))
#vectorize the train and test sets with the dictionary from Feature Extractor

def processXtrain_nobatching ():
    rawtX, tY = fe.readRawTrainingLines ()
    tX = fe.vectorizeLines(rawtX, ngramDict)
    return sortXtrain(tX, tY)

def processXtrain_nobatchingnosort ():
    rawtX, tY = fe.readRawTrainingLines ()
    tX = fe.vectorizeLines(rawtX, ngramDict)
    return tX, tY
    
def sortXtrain (rawtX, tY):
    sorted_lines = {0: [], 1: [], 2: [], 3: [], 4: []}
    for i in range(len(tY)):
            lang = tY[i]
            sorted_lines[lang].append(rawtX[i])
    return sorted_lines

def processXtest_nobatching():
    testX = fe.readRawTestingLines()
    testX = fe.vectorizeLines(testX, ngramDict)
    return testX

#the entire section below is for either vectorizing new sets or loading pickle files

if sort_test == True:
    processedXtest = processXtest_nobatching ()
    pickle_out = open("processedXtest.pkl","wb")
    pickle.dump(processedXtest, pickle_out)
    pickle_out.close()
    print ('Created processed testset')
else:
    processedXtest = pickle.load(open("processedXtest.pkl", "rb"))
    print ('Loaded processed testset')
        
if sort_train == True:
    processedXtrain = processXtrain_nobatching ()
    pickle_out = open("processedXtrain.pkl","wb")
    pickle.dump(processedXtrain, pickle_out)
    pickle_out.close()
    print ('Created processed trainset')
else:
    processedXtrain = pickle.load(open("processedXtrain.pkl", "rb"))
    print ('Loaded processed trainset')
    
if  raw_train == True:
    rawXtrain = processXtrain_nobatchingnosort ()
    pickle_out = open("processedXtrain_nosort.pkl","wb")
    pickle.dump(rawXtrain, pickle_out)
    pickle_out.close()
    print ('Created raw trainset')
else:
    rawXtrain = pickle.load(open("processedXtrain_nosort.pkl", "rb"))
    print ('Loaded raw trainset')    
    
#kmeans-clustering#######################################################################################################       

def cluster_kmeans (x, n):
    kmeans = KMeans(n_clusters = n, random_state = 0).fit(x)
    return kmeans
  
def cluster_lang (x):
    kmeans = cluster_kmeans (x, num_clust)   
    cluster_centers = kmeans.cluster_centers_
    print ('Computed clusters!')
    return cluster_centers

#generate cluster centers for each language
lang_0 = cluster_lang (processedXtrain[0])
lang_1 = cluster_lang (processedXtrain[1])
lang_2 = cluster_lang (processedXtrain[2])
lang_3 = cluster_lang (processedXtrain[3])       
lang_4 = cluster_lang (processedXtrain[4]) 

#combine everything into a training set, fetch the test set
train_x = []
train_y = []
test_x = processedXtest
raw_test_y = (pandas.read_csv('data/train_set_y.csv')['Category'])
test_x, val_test_x, raw_test_y, val_test_y = train_test_split(test_x, raw_test_y, test_size=0.01, random_state=42)

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

def predict(train_x, train_y, test, k):
    distances = []
    labels = []
    weights = []
    weighted_knn = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    #for each element of the train set: calculate distances to the test vector
    for i in range(len(train_x)):
        train = train_x[i]
        dist = fe.distanceEuclidean(test, train)
        distances.append([dist, i])
        #sort training set by distances
        distances = sorted(distances)
    #find closest neighbors
    for i in range(k):
        index = distances[i][1]
        #calculate the weight
        weight = (1/(distances[i][0]))
        #weight = 1
        labels.append(train_y[index])
        weights.append(weight)
    #add up class counts after adjusting weights
    for i in range(len(labels)):
        for n in weighted_knn.keys():
            if labels[i] == int(n): 
                weighted_knn[n] += weights[i]
    final = max(weighted_knn, key=weighted_knn.get)
    return final
     
def kNN(train_x, train_y, test_x, predicted, k):
    for i in range(len(test_x)):
        print(i)
        test = test_x[i]
        pr = predict(train_x, train_y, test, k)
        predicted.append(pr)
    return predicted
    
test_y = kNN(train_x, train_y, val_test_x, predicted, k)
#generate submission file
for i in range(len(test_y)):
    csvrow = []
    index = i
    prediction = test_y[i]
    csvrow = [index, prediction]
    print(csvrow)
    with open("submission.csv", "a", newline='') as outputFile:
        row = csv.writer(outputFile)
        row.writerow(csvrow)

#testing accuracy for validation
acc_pred = np.asarray(test_y)
print (accuracy_score(val_test_y, acc_pred))

