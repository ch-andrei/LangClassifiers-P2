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
maxLinesBeforeDictIsEmptied = 4096 # will build dictionaries in increments of this # RAM saver

########################################################################################################################

forceBuildNewDictionary = False
dictionaryDefaultPickleName = "dictionary.pkl"

# training dataset info
dataFolderName = "data/"
trainSetXFilename = "train_set_x.csv"
trainSetYFilename = "train_set_y.csv"

languageNames = {0: "slovak", 1: "french", 2: "spanish", 3: "german", 4: "polish"}

numEntriesMult = 1
numEntriesBase = 276516
maxNumEntries = numEntriesBase * numEntriesMult  # will read up to this many entries in the dataset

# training features extraction parameter
ngramMin = 1
ngramMax = 4

# produces a dictionary in the format:
# dictionary = {ngram1: [countInLang1, countInLang2, ..., countinLangN], ngram2: {...}, ...}

########################################################################################################################
# generic tools

def fileLinesCount(fname):
    with open(fname, "r", encoding="utf8") as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def getDictionaryName(numLinesParsed, ngramMinCount, ngramMaxCount):
    return "dict_{}-{}grams_{}Train.pkl".format(ngramMinCount, ngramMaxCount, numLinesParsed)

def checkForExistingDictionary(dictionaryFileName=dictionaryDefaultPickleName):
    return os.path.isfile(dataFolderName + dictionaryFileName)

def pickleReadOrWriteDictionary(dictionaryFileName=dictionaryDefaultPickleName, dictionary=None):
    if dictionary == None:
        if checkForExistingDictionary(dictionaryFileName):
            with open(dataFolderName + dictionaryFileName, "rb") as f:
                print("Reading dictionary pickle {}...".format(dataFolderName + dictionaryFileName))
                return pk.load(f)
    else:
        with open(dataFolderName + dictionaryFileName, "wb") as f:
            print("Dumping dictionary to pickle {}...".format(dataFolderName + dictionaryFileName))
            pk.dump(dictionary, f)

def getTrainXFileLineCount():
    return fileLinesCount(dataFolderName + trainSetXFilename)

def getTrainYFileLineCount():
    return fileLinesCount(dataFolderName + trainSetYFilename)

def getTrainXFile():
    return open(dataFolderName + trainSetXFilename, "r", encoding="utf8")

def getTrainYFile():
    return open(dataFolderName + trainSetYFilename, "r", encoding="utf8")

########################################################################################################################

def processTrainLine(line):
    # assuming line format: "id,value"
    line = line.strip()
    return line.split(',')

def processXLine(x):
    vals = processTrainLine(x)

    _ngrams = ng.toCharRangedNgramList(vals[1], ngramMin, ngramMax)

    ngrams = {}
    for ngram in _ngrams:
        if not ngram in ngrams:
            ngrams[ngram] = 1
        else:
            ngrams[ngram] += 1

    return ngrams

def processYLine(y):
    # y format: "id, class"
    vals = processTrainLine(y)
    return int(vals[1])

def processAddToDictionary(dictionary, x, y):
    ngrams = processXLine(x)
    label = processYLine(y)

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

    print("Process", i, "working on", num, "lines, starting at", process_startCount)

    numLinesProcessed = getTrainableData(q, maxTotalCount=maxNumEntries, maxCount=num + leftover, startCount=process_startCount)

    print("Process", i, "finished ", numLinesProcessed, "lines.")

def generateDictionary():
    maxLines = maxNumEntries
    totalLines = getTrainXFileLineCount() # - 1
    if (maxNumEntries > totalLines):
        maxLines = totalLines

    dictionaryFileName = getDictionaryName(maxLines, ngramMin, ngramMax)

    loadedDictionaryFromDisk = False
    dictionary = None
    if not forceBuildNewDictionary and checkForExistingDictionary(dictionaryFileName):
        dictionary = pickleReadOrWriteDictionary(dictionaryFileName)
        loadedDictionaryFromDisk = True
    else:
        global numProcesses

        q = mp.Manager().Queue()
        qCounts = mp.Manager().Queue()

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
                processes.append(Process(target=dataProcessTask, args=(q, qCounts, i, countPerProcess, leftover)))
            processes[i].start()

        for i in range(numProcesses):
            processes[i].join()

        del processes

        print ("########################################")
        print ("Collecting the generated dictionaries...")

        dictCount = 0
        totalDictCount = q.qsize()
        dictionary = {}
        while not q.empty():
            dict = q.get_nowait()
            for ngram, counts in dict.items():
                if ngram in dictionary:
                    dictionary[ngram] += counts
                else:
                    dictionary[ngram] = counts

            dictCount += 1
            print("\rProcessed {}/{} dictionaries.".format(dictCount, totalDictCount), end=" ")

        totalLinesProcessed = 0
        while not qCounts.empty():
            totalLinesProcessed += qCounts.get_nowait()

        print("Merge complete.")
        del q, qCounts

    if dictionary == None:
        print ("Could not generate or read a dictionary... Exiting.")
    elif loadedDictionaryFromDisk:
        print ("Loaded dictionary from disk.")
    else:
        print ("Generated new dictionary.")

    print("Computing statistics...")
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

    if not loadedDictionaryFromDisk:
        pickleReadOrWriteDictionary(dictionaryFileName, dictionary)

    return languageNames, counts, dictionary

def sortBestFeatures(dictionary):
    return sorted(dictionary.items(), key=lambda item: ngramImportanceHeuristic(item[0], item[1]), reverse=True)

def ngramImportanceHeuristic(ngram, counts, alpha = 0.1, lengthInfluence=0.01, uniquenessInfluence=0.01, n=len(languageNames)):
    uniquenessFactor = 1 - uniquenessMeasure(counts, n) * 2 / 3.14
    lengthFactor = np.sqrt(len(ngram))
    return np.log(counts.sum()) * (alpha + lengthInfluence * lengthFactor + uniquenessInfluence * uniquenessFactor)

def distanceEuclidean(a, b):
    return np.sqrt(((a-b)**2).sum())

def cosineAngle(a, b):
    return a.dot(b) / np.sqrt((a*a).sum() * (b*b).sum())

def uniquenessMeasure(a, n, useCosineAngle=True):
    b = np.ones(n) / n
    c = a / a.sum()
    if useCosineAngle:
        return cosineAngle(c,b)
    else:
        return distanceEuclidean(c, b)

def processDictionaryAsTrainingFeatures():
    languageNames, counts, dictionary = generateDictionary()

    bestNgrams = sortBestFeatures(dictionary)[:10000]

    del dictionary

    ngramsList = []
    ngramDict = {}
    index = 0
    ngramLangCounts = np.zeros(len(languageNames))
    for ngram, counts in bestNgrams:
        # print ("[{}] {}".format(ngram, ngramImportanceHeuristic(ngram, counts)), counts.sum(), counts)
        ngramsList.append(ngram)
        ngramDict[ngram] = (counts, index)
        ngramLangCounts += counts
        index += 1

    return ngramsList, ngramDict, ngramLangCounts

def vectorizeLine(line, ngrams):
    _ngrams = processXLine(line)

    vector = np.zeros(len(ngrams), np.uint16)

    for ngram in _ngrams:
        if ngram in ngrams:
            vector[ngrams[ngram][1]] += 1
            print (ngram, ngrams[ngram][1], ngrams[ngram])

    return vector


def main():
    processDictionaryAsTrainingFeatures()

if __name__ == "__main__":
    main()
