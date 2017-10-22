import sys, operator, os
import pickle as pk
import multiprocessing as mp
from multiprocessing import Process

import numpy as np

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

forceBuildNewDictionary = True
forceBuildNewBestDictionary = forceBuildNewDictionary or True

########################################################################################################################

dictionaryDefaultPickleName = "dictionary.pkl"

# training dataset info
dataFolderName = "data/"
trainSetXFilename = "mergedDatasetX.csv"
trainSetYFilename = "mergedDatasetY.csv"
# trainSetXFilename = "generatedTestSetX-500000.csv"
# trainSetYFilename = "generatedTestSetY-500000.csv"

# test data set
fakeTrainSetXFilename = "generatedTestSetX-100000.csv"
fakeTrainSetYFilename = "generatedTestSetY-100000.csv"
testSetXFilename = "test_set_x.csv"

maxNumEntries = 500000 # maximum number of lines to read

########################################################################################################################

languageNames = {0: "slovak", 1: "french", 2: "spanish", 3: "german", 4: "polish"}

# training features extraction parameter
ngramMin = 1
ngramMax = 1

MAX_FEATURE_DIM=1000 # up to how many feature to keep

# produces a dictionary in the format:
# dictionary = {ngram1: [countInLang1, countInLang2, ..., countinLangN], ngram2: {...}, ...}

########################################################################################################################
# generic helper functions

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

# given a list of labels, count the distribution of labels
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
    line = line.replace(" ", "")
    return line.split(',')

# compute ngrams for an x line
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

# get the label value
def processYLine(y):
    # y format: "id, class"
    vals = processTrainLine(y)
    return int(vals[1])

# process an X-Y line pair and add resulting ngrams to an existing dictionary
def processAddToDictionary(dictionary, x, y):
    ngrams = processXLine(x)
    label = processYLine(y)

    for ngram, count in ngrams.items():
        if ngram in dictionary:
            dictionary[ngram][label] += count
        else:
            dictionary[ngram] = np.zeros(len(languageNames), np.uint32)
            dictionary[ngram][label] = count

# read the training data files, add processed ngrams to dictionary, append dictionary to a queue
# queue is used for thread synchronization issues
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

# function to process train files by a Python process
def processTrainingDataToDictionaryTask(q, qCounts, i, num, leftover=0):
    process_startCount = i * num

    print("Process", i, "working on", num, "lines, starting at", process_startCount)

    numLinesProcessed = processTrainingDataToDictionary(q, maxTotalCount=maxNumEntries, maxCount=num + leftover, startCount=process_startCount)

    print("Process", i, "finished ", numLinesProcessed, "lines.")

# generates a dictionary or reads it from a.pkl file provided one was previously generated
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

# read TRAINING x,y lines without processing (simply makes lists of X and Y entries) # repeating code for lack of time
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

# read TESTING x lines without processing (simply makes a list of X entries) # repeating code for lack of time
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

# sorts dictionary to a list of tuples in descending order of ngram importance (best first)
# returns up to featureDim best ngrams (throws away the rest)
def sortBestFeatures(dictionary, featureDim):
    # dictionary {ngram: counts, ...}
    sortedFeatures = sorted(dictionary.items(), key=lambda item: ngramImportanceHeuristic(item[0], item[1]), reverse=True)

    bestFeatures = []
    for ngram, counts in sortedFeatures:
        bestFeatures.append((ngram, counts))

    return bestFeatures[:featureDim]

# estimates ngrams importance based on its length and uniqueness
def ngramImportanceHeuristic(ngram, counts, alpha = 1, lengthInfluence=0.01, uniquenessInfluence=0.01, n=len(languageNames), scaleByCounts=True):
    uniquenessFactor = 1 - uniquenessMeasure(counts, n)
    lengthFactor = np.sqrt(len(ngram))
    h = lengthInfluence * lengthFactor + uniquenessInfluence * uniquenessFactor
    if scaleByCounts:
        h *= (alpha + np.log(counts.sum()))
    return h

def distanceEuclidean(a, b):
    return np.sqrt(((a-b)**2).sum())

def cosineAngle(a, b):
    return a.dot(b) / np.sqrt((a*a).sum() * (b*b).sum())

# estimates the uniquness of an ngram based on the distance of its counts vector to the least unique vector
# ex: for 2-dimension: [1,1] is the direction of the least unique vector, [0,1] and [1,0] are the most unique vectors
def uniquenessMeasure(a, n, useCosineAngle=True):
    b = np.ones(n) / np.sqrt(n)
    c = a / a.sum()
    if useCosineAngle:
        return cosineAngle(c,b) * 2 / 3.14
    else:
        return distanceEuclidean(c, b)

# read or generate a dictionary given the training files
# returns a tuple representing:
# (a list of best ngrams, ngram dictionary {ngram: counts}, total ngram counts for each language)
def processDictionaryAsTrainingFeatures(maxFeatureDim=MAX_FEATURE_DIM,
                                        force_recompute_best=forceBuildNewBestDictionary,
                                        force_recompute_dict=forceBuildNewDictionary):
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

    # for ngram, count in ngramDict.items():
    #     print(ngram, count)

    return ngramsList, ngramDict, ngramLangCounts

# returns a list of vectors representing raw X lines given an ngram dictionary
def vectorizeLines(lines, ngramDict):
    vectors = []
    for line in lines:
        vector = vectorizeLine(line, ngramDict)
        vectors.append(vector)
    return vectors

# returns a vector representing a single line given an ngram dictionary
def vectorizeLine(line, ngramDict):
    lineNgrams = processXLine(line)

    vector = np.zeros(len(ngramDict), np.float32)

    ngrams = []
    for ngram, count in lineNgrams.items():
        if ngram in ngramDict:
            ngrams.append(ngram)
            vector[ngramDict[ngram][1]] += count

    return vector

# call to generate the dictionary and the best features (saves to .pkl files)
def main():
    processDictionaryAsTrainingFeatures()

if __name__ == "__main__":
    main()
