import csv
import sys

import numpy as np

import ngramGenerator as ng

dataFolderName = "data/"
trainSetXFilename = "train_set_x.csv"
trainSetYFilename = "train_set_y.csv"

numFeaturesMultiplier = 1
numFeaturesBase = 4
maxNumFeatures = numFeaturesBase * numFeaturesMultiplier

def getTrainXFile():
    return open(dataFolderName + trainSetXFilename, "r", encoding="utf8")

def getTrainYFile():
    return open(dataFolderName + trainSetYFilename, "r", encoding="utf8")

def processTrainLine(line):
    # assuming line format: "id,value"
    line = line.strip()
    return line.split(',')

def processTrainXLine(x):
    vals = processTrainLine(x)

    # TODO
    # anything special to processing the raw text from the training set
<<<<<<< HEAD
    out = vals
=======
    out = ng.toCharRangedNgramList(vals[1], 2, 3)
>>>>>>> 190a416c17647fffab932ec46da34101a8e835ed

    return out

def processTrainYLine(y):
    # y format: "id, class"
<<<<<<< HEAD
    val = processTrainLine(y)

    # TODO
    out = val
=======
    vals = processTrainLine(y)

    # TODO
    out = int(vals[1])
>>>>>>> 190a416c17647fffab932ec46da34101a8e835ed

    return out

def getTrainableData(maxCount=maxNumFeatures):
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

<<<<<<< HEAD
                tX.append(processTrainXLine(_tXval)[1])
                tY.append(int(processTrainYLine(_tYval)[1]))
=======
                tX.append(processTrainXLine(_tXval))
                tY.append(processTrainYLine(_tYval))

                # TODO
                # generate ngrams from the cleaned words
>>>>>>> 190a416c17647fffab932ec46da34101a8e835ed

                count += 1
            except:
                print("Unexpected error:", sys.exc_info()[0])

    return tX, tY, count

def main():
<<<<<<< HEAD
    # tX, tY, numFeatures = getTrainableData()
    word = "i like turtles"
    print(ng.toWordRangedNgramList(word, 2))
    print(ng.toCharRangedNgramList(word, 5))
=======
    tX, tY, numFeatures = getTrainableData()

    print ("Processed", numFeatures, "entries from the training data.")
    # andrei: u can see the contents of the generated list here
    for tx, ty in zip(tX, tY):
        print (ty, tx)


>>>>>>> 190a416c17647fffab932ec46da34101a8e835ed

if __name__ == "__main__":
    main()










