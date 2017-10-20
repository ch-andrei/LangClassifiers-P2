import sys, operator, os
import pickle as pk
import multiprocessing as mp
from multiprocessing import Process

import numpy as np
import scipy.sparse as scsp

import ngramGenerator as ng

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
forceBuildNewBestDictionary = False

dictionaryDefaultPickleName = "dictionary.pkl"

# training dataset info
dataFolderName = "data/"
trainSetXFilename = "train_set_x.csv"
trainSetYFilename = "train_set_y.csv"
fakeTrainSetXFilename = "generatedTestSetX-100000.csv"
fakeTrainSetYFilename = "generatedTestSetY-100000.csv"
testSetXFilename = "test_set_x.csv"

languageNames = {0: "slovak", 1: "french", 2: "spanish", 3: "german", 4: "polish"}

maxNumEntries = 500000

# training features extraction parameter
ngramMin = 1
ngramMax = 3

# produces a dictionary in the format:
# dictionary = {ngram1: [countInLang1, countInLang2, ..., countinLangN], ngram2: {...}, ...}

########################################################################################################################
# generic tools

def checkForExistingDataFile(filename):
    return os.path.isfile(dataFolderName + filename)

def pickleReadOrWriteObject(filename=dictionaryDefaultPickleName, object=None):
    if object == None:
        if checkForExistingDataFile(filename):
            with open(dataFolderName + filename, "rb") as f:
                print("Reading pickle [{}]...".format(dataFolderName + filename))
                return pk.load(f)
        return None
    else:
        with open(dataFolderName + filename, "wb") as f:
            print("Dumping to pickle [{}]...".format(dataFolderName + filename))
            pk.dump(object, f)

def fileLinesCount(fname):
    with open(fname, "r", encoding="utf8") as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def getDictionaryName(numLinesParsed, ngramMinCount, ngramMaxCount):
    return "dict_{}-{}grams_{}Train.pkl".format(ngramMinCount, ngramMaxCount, numLinesParsed)

def getTrainXFileLineCount(useFakeTrainSet=False):
    if useFakeTrainSet:
        return fileLinesCount(dataFolderName + fakeTrainSetXFilename)
    return fileLinesCount(dataFolderName + trainSetXFilename)

def getTrainYFileLineCount():
    return fileLinesCount(dataFolderName + trainSetYFilename)

def getTestXFileLineCount():
    return fileLinesCount(dataFolderName + testSetXFilename)

def getTrainFiles(useFakeTrainSet=False):
    if useFakeTrainSet:
        return getFakeTrainXFile(), getFakeTrainYFile()
    return getTrainXFile(), getTrainYFile()

def getFakeTrainXFile():
    return open(dataFolderName + fakeTrainSetXFilename, "r", encoding="utf8")

def getFakeTrainYFile():
    return open(dataFolderName + fakeTrainSetYFilename, "r", encoding="utf8")

def getTrainXFile():
    return open(dataFolderName + trainSetXFilename, "r", encoding="utf8")

def getTrainYFile():
    return open(dataFolderName + trainSetYFilename, "r", encoding="utf8")

def getTestXFile():
    return open(dataFolderName + testSetXFilename, "r", encoding="utf8")

########################################################################################################################

def getTrainClassWeights(labels):
    trainYCounts = np.zeros(len(languageNames), np.uint32)
    for label in labels:
        try:
            trainYCounts[label] += 1
        except:
            continue
    return trainYCounts

def processTrainLine(line):
    # assuming line format: "id,value"
    line = line.strip()
    return line.split(',')

def processXLine(x):
    vals = processTrainLine(x)

    _ngrams = ng.toCharRangedNgramList(vals[1], ngramMin, ngramMax)

    ngrams = {}
    for ngram in _ngrams:
        if " " in ngram:
            continue
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

def processTrainingDataToDictionary(q, maxTotalCount=maxNumEntries, maxCount=maxNumEntries, startCount=0):
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
                break

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
                break

            if lineCount % maxLinesBeforeDictIsEmptied == 0:
                print (lineCount)
                q.put(dictionary)
                dictionary = {}

    q.put(dictionary)

    return lineCount

def processTrainingDataToDictionaryTask(q, qCounts, i, num, leftover=0):
    process_startCount = i * num

    print("Process", i, "working on", num, "lines, starting at", process_startCount)

    numLinesProcessed = processTrainingDataToDictionary(q, maxTotalCount=maxNumEntries, maxCount=num + leftover, startCount=process_startCount)

    print("Process", i, "finished ", numLinesProcessed, "lines.")

def generateDictionary(force_recompute=forceBuildNewDictionary):
    maxLines = maxNumEntries
    totalLines = getTrainXFileLineCount()
    if (maxNumEntries > totalLines):
        maxLines = totalLines

    dictionaryFileName = getDictionaryName(maxLines, ngramMin, ngramMax)

    loadedDictionaryFromDisk = False
    dictionary = None
    if not force_recompute:
        dictionary = pickleReadOrWriteObject(dictionaryFileName)
        loadedDictionaryFromDisk = True
    else:
        global numProcesses

        print("Recomputing dictionary...")

        q = mp.Manager().Queue()
        qCounts = mp.Manager().Queue()

        countPerProcess = int(maxLines / numProcesses)
        if (countPerProcess < minLinesPerProcess):
            countPerProcess = maxLines
            numProcesses = 1

        processes = []
        for i in range(numProcesses):
            if (i < numProcesses - 1):
                processes.append(Process(target=processTrainingDataToDictionaryTask, args=(q, qCounts, i, countPerProcess)))
            else:
                leftover = maxLines - countPerProcess * numProcesses
                processes.append(Process(target=processTrainingDataToDictionaryTask, args=(q, qCounts, i, countPerProcess, leftover)))
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
    count = 0
    counts = np.zeros(5, np.uint32)
    for ngram, _counts in dictionary.items():
        totalCount += _counts.sum()
        counts += _counts

        unique = len(np.where(_counts > 0)[0]) == 1
        if unique:
            uniqueCount += 1
        count += 1

    print (languageNames)
    print (counts)
    print ("totalCount", totalCount, "; countNgrams", count, ", uniqueCount", uniqueCount, ", uniqueness ratio ", uniqueCount/count)

    if not loadedDictionaryFromDisk:
        pickleReadOrWriteObject(dictionaryFileName, dictionary)

    return languageNames, counts, dictionary

def readRawTrainingLines(maxTotalCount=maxNumEntries, maxCount=maxNumEntries, startCount=0, useFakeTrainSet=False):
    rawtX = []
    tY = []

    # read files
    lineCount = 0
    tXfile, tYfile = getTrainFiles(useFakeTrainSet)

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
            break

    for tXval, tYval in zip(tXfile, tYfile):
        try:
            if lineCount >= maxCount or startCount + lineCount >= maxTotalCount:
                # reached max number of features
                break

            rawtX.append(tXval)
            tY.append(processYLine(tYval))

            lineCount += 1

        except ValueError:
            print("Unexpected error:", sys.exc_info()[0])
            continue
        except StopIteration:
            # end of file
            break

    tXfile.close()
    tYfile.close()

    return rawtX, tY

def readRawTestingLines(maxTotalCount=maxNumEntries, maxCount=maxNumEntries, startCount=0):
    rawtX = []

    # read files
    lineCount = 0
    with getTestXFile() as tXfile:

        # skip the header line
        next(tXfile)

        skipCount = 0
        while (skipCount < startCount):
            try:
                next(tXfile)
                skipCount += 1
            except StopIteration:
                # end of file
                break

        for tXval in tXfile:
            try:
                if lineCount >= maxCount or startCount + lineCount >= maxTotalCount:
                    # reached max number of features
                    break

                rawtX.append(tXval)

                lineCount += 1

            except ValueError:
                print("Unexpected error:", sys.exc_info()[0])
                continue
            except StopIteration:
                # end of file
                break

    return rawtX

def sortBestFeatures(dictionary, featureDim):
    # dictionary {ngram: counts, ...}
    sortedFeatures = sorted(dictionary.items(), key=lambda item: ngramImportanceHeuristic(item[0], item[1]), reverse=True)

    bestFeatures = []
    for ngram, counts in sortedFeatures:
        bestFeatures.append((ngram, counts))

    return bestFeatures[:featureDim]

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

def processDictionaryAsTrainingFeatures(maxFeatureDim=10000, force_recompute_best=forceBuildNewBestDictionary, force_recompute_dict=forceBuildNewDictionary):
    bestFeaturesDictFilename = "ngramListDictCounts_max{}_{}-{}.pkl".format(maxFeatureDim, ngramMin, ngramMax)

    ngramListDictCounts = pickleReadOrWriteObject(bestFeaturesDictFilename)

    if ngramListDictCounts == None or force_recompute_best:
        print("Recomputing best features...")

        languageNames, counts, dictionary = generateDictionary(force_recompute=force_recompute_dict)

        bestNgrams = sortBestFeatures(dictionary, maxFeatureDim)

        del dictionary

        ngramsList = []
        ngramDict = {}
        ngramLangCounts = np.zeros(len(languageNames))
        for index, (ngram, counts) in enumerate(bestNgrams):
            # print ("[{}] {}".format(ngram, ngramImportanceHeuristic(ngram, counts)), counts.sum(), counts)
            ngramsList.append(ngram)
            ngramDict[ngram] = (counts, index)
            ngramLangCounts += counts

        ngramListDictCounts = (ngramsList, ngramDict, ngramLangCounts)

        pickleReadOrWriteObject(bestFeaturesDictFilename, ngramListDictCounts)

    ngramsList, ngramDict, ngramLangCounts = ngramListDictCounts

    return ngramsList, ngramDict, ngramLangCounts

def vectorizeLines(lines, ngramDict):
    vectors = []
    for line in lines:
        vector = vectorizeLine(line, ngramDict)
        vectors.append(vector)
    return vectors

def vectorizeLine(line, ngramDict):
    lineNgrams = processXLine(line)

    return vectorizeNgrams(lineNgrams, ngramDict)

def vectorizeNgrams(lineNgrams, ngramDict):
    vector = np.zeros(len(ngramDict), np.float)

    for ngram, count in lineNgrams.items():
        if ngram in ngramDict:
            vector[ngramDict[ngram][1]] += count

    # return scsp.csr_matrix(vector)
    return vector

def printNgramVector(vector, ngramsList):
    print("Printing vector:")
    _, indices, counts = scsp.find(vector)
    for index, count in zip(indices, counts):
        print("[{}]".format(ngramsList[index]), "index", index, "count:", count)

def main():
    processDictionaryAsTrainingFeatures()

if __name__ == "__main__":
    main()
