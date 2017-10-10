import csv
import sys
import operator
import json

import numpy as np

import ngramGenerator as ng

# training dataset info
dataFolderName = "data/"
trainSetXFilename = "train_set_x.csv"
trainSetYFilename = "train_set_y.csv"

outputFilename = dataFolderName + "languages.json"

numLanguages = 5
languageNames = {0: "slovak", 1: "french", 2: "spanish", 3: "german", 4: "polish"}

numEntriesMult = 1
numEntriesBase = 64
maxNumEntries = numEntriesBase * numEntriesMult  # will read up to this many entries in the dataset

# training features extraction parameter
ngramMin = 3
ngramMax = 10

###########
# functions
###########

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

    # TODO
    # anything special to processing the raw text from the training set
    ngrams = ng.toCharRangedNgramList(vals[1], ngramMin, ngramMax)

    frequencies = {}
    for ngram in ngrams:
        if not ngram in frequencies:
            frequencies[ngram] = 1
        else:
            frequencies[ngram] += 1

    return frequencies

def processTrainYLine(y):
    # y format: "id, class"
    vals = processTrainLine(y)

    # TODO
    out = int(vals[1])

    return out

def getTrainableData(maxCount=maxNumEntries):
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
                if (count % 10000 == 0):
                    print(count)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                continue

    return tX, tY, count

def main():
    tX, tY, numFeatures = getTrainableData()

    # print ("Processed", numFeatures, "entries from the training data.")
    # # andrei: u can see the contents of the generated list here
    # for frequencies, label in zip(tX, tY):
    #     print (label, getFrequenciesAsSortedTupleList(frequencies))
    #     # using getFrequenciesAsSortedTupleList(frequencies) converts dictionary to a list of tuples; cant sort a dictionary

    languages = { language: {} for label, language in languageNames.items()}
    for frequencies, label in zip(tX, tY):
        language = languages[languageNames[label]]
        for ngram, count in frequencies.items():
            if not ngram in language:
                language[ngram] = frequencies[ngram]
            else:
                language[ngram] += frequencies[ngram]

    # for languageName, languageNgrams in languages.items():
    #     print (languageName)
    #     print (getFrequenciesAsSortedTupleList(languageNgrams)[:100])

    with open(outputFilename, "wb") as f:
        f.write(str(languages).encode('utf8'))

if __name__ == "__main__":
    main()










