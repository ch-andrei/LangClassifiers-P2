import sys
from collections import Counter
import time

import ngramGenerator as ng


def getTrainXFile(dataFolderName,trainSetXFilename):
    return open(dataFolderName + trainSetXFilename, "r")#, encoding="utf8")

def getTrainYFile(dataFolderName,trainSetYFilename):
    return open(dataFolderName + trainSetYFilename, "r")#, encoding="utf8")

def getTestFile(dataFolderName,testFilename):
    return open(dataFolderName + testFilename, "r")#, encoding="utf8")

def processTrainLine(line):
    # assuming line format: "id,value"
    line = line.strip()
    return line.split(',')

def processTrainXLine(x, MIN_GRAM, MAX_GRAM):
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

def getTrainableData(maxCount, dataFolderName,trainSetXFilename, trainSetYFilename, MIN_GRAM, MAX_GRAM):
    tX = []
    tY = []

    # read files
    count = 0
    with getTrainXFile(dataFolderName,trainSetXFilename) as tXfile, getTrainYFile(dataFolderName,trainSetYFilename) as tYfile:

        # skip the header line
        next(tXfile)
        next(tYfile)

        for _tXval, _tYval in zip(tXfile, tYfile):
            try:
                if count >= maxCount:
                    # reached max number of features
                    break

                tX.append(processTrainXLine(_tXval.lower(),MIN_GRAM, MAX_GRAM ))
                tY.append(processTrainYLine(_tYval.lower()))

                # TODO
                # generate ngrams from the cleaned words

                count += 1
            except:
                print("Unexpected error:", sys.exc_info()[0])

    return tX, tY, count

def generate_freq(NUM_SAMPLES, dataFolderName,trainSetXFilename, trainSetYFilename, MIN_GRAM, MAX_GRAM):
    start_time = time.time()
    tX, tY, numFeatures = getTrainableData(NUM_SAMPLES,dataFolderName,trainSetXFilename, trainSetYFilename, MIN_GRAM, MAX_GRAM)
    print('Generate N gram time: ', time.time() - start_time)
    print("Processed", numFeatures, "entries from the training data.")
    # andrei: u can see the contents of the generated list here

    start_time = time.time()
    processed_data = []

    for tx, ty in zip(tX, tY):
        freq = dict(Counter(tx))
        #freq['__CATEGORY__'] = ty
        processed_data.append([freq, ty])
        #total_counter += freq
    print('Count ngram time: ', time.time() - start_time)
    print('len data = ',len(processed_data))
    return processed_data


def getTestData(dataFolderName,testFilename,MIN_GRAM, MAX_GRAM):
    testlines = []
    with getTestFile(dataFolderName,testFilename) as testfile:
        # skip the header line
        next(testfile)
        for _tXval in testfile:
            try:
                testlines.append(processTrainXLine(_tXval.lower(), MIN_GRAM, MAX_GRAM))
            except:
                print("Unexpected error:", sys.exc_info()[0])
    test_freq = []
    for t in testlines:
        freq = dict(Counter(t))
        test_freq.append(freq)

    return test_freq












