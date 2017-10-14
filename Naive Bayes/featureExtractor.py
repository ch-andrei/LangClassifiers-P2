import csv
import sys

import numpy as np

from collections import Counter
import time

import ngramGenerator as ng

dataFolderName = "data/"
trainSetXFilename = "train_set_x.csv"
trainSetYFilename = "train_set_y.csv"
testFilename = 'test_set_x.csv'

numFeaturesMultiplier = 1
NUM_SAMPLES = 276516
maxNumSamples = NUM_SAMPLES * numFeaturesMultiplier

MIN_GRAM = 1
MAX_GRAM = 3

def getTrainXFile():
    return open(dataFolderName + trainSetXFilename, "r")#, encoding="utf8")

def getTrainYFile():
    return open(dataFolderName + trainSetYFilename, "r")#, encoding="utf8")

def getTestFile():
    return open(dataFolderName + testFilename, "r")#, encoding="utf8")

def processTrainLine(line):
    # assuming line format: "id,value"
    line = line.strip()
    return line.split(',')

def processTrainXLine(x):
    vals = processTrainLine(x)

    # TODO
    # anything special to processing the raw text from the training set
    out = ng.toCharRangedNgramList(vals[1], MIN_GRAM, MAX_GRAM)

    return out

def processTrainYLine(y):
    # y format: "id, class"
    vals = processTrainLine(y)

    # TODO
    out = int(vals[1])

    return out

def getTrainableData(maxCount=maxNumSamples):
    tX = []
    tY = []

    # read files
    count = 0
    with getTrainXFile() as tXfile, getTrainYFile() as tYfile:

        # skip the header line
        next(tXfile)
        next(tYfile)

        for _tXval, _tYval in zip(tXfile, tYfile):
            try:
                if count >= maxCount:
                    # reached max number of features
                    break

                tX.append(processTrainXLine(_tXval))
                tY.append(processTrainYLine(_tYval))

                # TODO
                # generate ngrams from the cleaned words

                count += 1
            except:
                print("Unexpected error:", sys.exc_info()[0])

    return tX, tY, count

def generate_freq():
    start_time = time.time()
    tX, tY, numFeatures = getTrainableData()
    print 'Generate N gram time: ', time.time() - start_time
    print ("Processed", numFeatures, "entries from the training data.")
    # andrei: u can see the contents of the generated list here

    start_time = time.time()
    processed_data = []

    for tx, ty in zip(tX, tY):
        freq = dict(Counter(tx))
        #freq['__CATEGORY__'] = ty
        processed_data.append([freq, ty])
        #total_counter += freq
    print 'Count ngram time: ', time.time() - start_time
    print 'len data = ',len(processed_data)
    return processed_data


def getTestData():
    testlines = []
    with getTestFile() as testfile:
        # skip the header line
        next(testfile)
        for _tXval in testfile:
            try:
                testlines.append(processTrainXLine(_tXval))
            except:
                print("Unexpected error:", sys.exc_info()[0])
    test_freq = []
    for t in testlines:
        freq = dict(Counter(t))
        test_freq.append(freq)

    return test_freq


if __name__ == "__main__":
    generate_freq()










