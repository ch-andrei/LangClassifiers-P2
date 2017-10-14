import sys
import operator
import os

import multiprocessing as mp
from multiprocessing import Process

import numpy as np

import ngramGenerator as ng

import pickle as pk

########################################################################################################################

useAllCpus = True
if useAllCpus:
    numProcesses = mp.cpu_count()
else:
    numProcesses = max(1, int(mp.cpu_count() / 2)) # will use only half of the available CPUs

minLinesPerProcess = 1024 # if less than this, will only use 1 CPU
maxLinesBeforeDictIsEmptied = 4096 # will build dictionaries in increments of this

########################################################################################################################

forceBuildNewDictionary = False
dictionaryPickleName = "dictionary.pkl"

# training dataset info
dataFolderName = "data/"
trainSetXFilename = "train_set_x.csv"
trainSetYFilename = "train_set_y.csv"

outputFilename = dataFolderName + "languages.json"

languageNames = {0: "slovak", 1: "french", 2: "spanish", 3: "german", 4: "polish"}

numEntriesMult = 1024
numEntriesBase = 1024
maxNumEntries = numEntriesBase * numEntriesMult  # will read up to this many entries in the dataset

# training features extraction parameter
ngramMin = 1
ngramMax = 6

########################################################################################################################

def file_len(fname):
    with open(fname, "r", encoding="utf8") as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def getTrainXFileLineCount():
    return file_len(dataFolderName + trainSetXFilename)

def getTrainYFileLineCount():
    return file_len(dataFolderName + trainSetYFilename)

def getTrainXFile():
    return open(dataFolderName + trainSetXFilename, "r", encoding="utf8")

def getTrainYFile():
    return open(dataFolderName + trainSetYFilename, "r", encoding="utf8")

def processTrainLine(line):
    # assuming line format: "id,value"
    line = line.strip()
    return line.split(',')

def getFrequenciesAsSortedTupleList(frequencies):
    return sorted(frequencies.items(), key=operator.itemgetter(1), reverse=True)

def processTrainXLine(x):
    vals = processTrainLine(x)

    _ngrams = ng.toCharRangedNgramList(vals[1], ngramMin, ngramMax)

    ngrams = {}
    for ngram in _ngrams:
        if not ngram in ngrams:
            ngrams[ngram] = 1
        else:
            ngrams[ngram] += 1

    return ngrams

def processTrainYLine(y):
    # y format: "id, class"
    vals = processTrainLine(y)
    return int(vals[1])

def processAddToDictionary(dictionary, x, y):
    ngrams = processTrainXLine(x)
    label = processTrainYLine(y)

    for ngram, count in ngrams.items():
        if ngram in dictionary:
            dictionary[ngram][label] += count
        else:
            dictionary[ngram] = np.zeros(len(languageNames), np.uint32)
            dictionary[ngram][label] = count

def getTrainableData(q, maxTotalCount=maxNumEntries, maxCount=maxNumEntries, startCount=0):
    dictionary = {}

    # read files
    lineCount = 0
    with getTrainXFile() as tXfile, getTrainYFile() as tYfile:

        # skip the header line
        next(tXfile)
        next(tYfile)

        skipCount = 0
        while (skipCount < startCount):
            try:
                next(tXfile)
                next(tYfile)
                skipCount += 1
            except StopIteration:
                # end of file
                return

        for _tXval, _tYval in zip(tXfile, tYfile):
            try:
                if lineCount >= maxCount or startCount + lineCount >= maxTotalCount:
                    # reached max number of features
                    break

                processAddToDictionary(dictionary, _tXval, _tYval)

                lineCount += 1

            except ValueError:
                print("Unexpected error:", sys.exc_info()[0])
                continue
            except StopIteration:
                # end of file
                return

            if lineCount % maxLinesBeforeDictIsEmptied == 0:
                print (lineCount)
                q.put(dictionary)
                dictionary = {}

    q.put(dictionary)

    return lineCount

def dataProcessTask(q, qCounts, i, num, leftover=0):
    process_startCount = i * num

    print("process", i, "working on", num, "lines, starting at", process_startCount)

    numLinesProcessed = getTrainableData(q, maxTotalCount=maxNumEntries, maxCount=num + leftover, startCount=process_startCount)

    print("process", i, "got ", numLinesProcessed, "lines.")

def main():

    loadedDictionaryFromDisk = False
    dictionary = None
    if checkForExistingDictionary() and not forceBuildNewDictionary:
        dictionary = pickleReadOrWriteDictionary()
        loadedDictionaryFromDisk = True
    else:
        global numProcesses

        q = mp.Manager().Queue()
        qCounts = mp.Manager().Queue()

        maxLines = maxNumEntries
        totalLines = getTrainXFileLineCount()
        if (maxNumEntries > totalLines):
            maxLines = totalLines

        countPerProcess = int(maxLines / numProcesses)
        if (countPerProcess < minLinesPerProcess):
            countPerProcess = maxLines
            numProcesses = 1

        processes = []
        for i in range(numProcesses):
            if (i < numProcesses - 1):
                processes.append(Process(target=dataProcessTask, args=(q, qCounts, i, countPerProcess)))
            else:
                leftover = maxLines - countPerProcess * numProcesses
                print("leftover", leftover)
                processes.append(Process(target=dataProcessTask, args=(q, qCounts, i, countPerProcess, leftover)))
            processes[i].start()

        for i in range(numProcesses):
            processes[i].join()

        del processes

        totalLinesProcessed = 0
        dictionary = {}
        while not q.empty():
            print("getting an item from queue...")
            dict = q.get_nowait()
            for ngram, counts in dict.items():
                if ngram in dictionary:
                    dictionary[ngram] += counts
                else:
                    dictionary[ngram] = counts

        while not qCounts.empty():
            totalLinesProcessed += qCounts.get_nowait()

        del q, qCounts

    print("computing statistics...")
    # compute some statistics
    uniqueCount = 0
    totalCount = 0
    counts = np.zeros(5, np.uint32)
    for ngram, _counts in dictionary.items():
        totalCount += _counts.sum()
        counts += _counts

        unique = len(np.where(_counts > 0)[0]) == 1
        if unique:
            uniqueCount += 1

    print (languageNames)
    print (counts)
    print ("totalCount", totalCount, ", uniqueCount", uniqueCount, ", uniqueness ratio ", uniqueCount/totalCount)

    # for languageName, languageNgrams in languages.items():
    #     print (languageName)
    #     print (getFrequenciesAsSortedTupleList(languageNgrams)[:100])

    if not loadedDictionaryFromDisk:
        pickleReadOrWriteDictionary(dictionary)

def checkForExistingDictionary():
    return os.path.isfile(dataFolderName + dictionaryPickleName)

def pickleReadOrWriteDictionary(dictionary=None):
    if dictionary == None:
        with open(dataFolderName + dictionaryPickleName, "rb") as f:
            print("Reading dictionary pickle...")
            return pk.load(f)
    else:
        with open(dataFolderName + dictionaryPickleName, "wb") as f:
            print("Dumping dictionary pickle...")
            pk.dump(dictionary, f)

if __name__ == "__main__":
    main()
