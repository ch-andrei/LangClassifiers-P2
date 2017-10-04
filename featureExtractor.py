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
    out = ng.toCharRangedNgramList(vals[1], 2, 3)

    return out

def processTrainYLine(y):
    # y format: "id, class"
    vals = processTrainLine(y)

    # TODO
    out = int(vals[1])

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

                tX.append(processTrainXLine(_tXval))
                tY.append(processTrainYLine(_tYval))

                # TODO
                # generate ngrams from the cleaned words

                count += 1
            except:
                print("Unexpected error:", sys.exc_info()[0])

    return tX, tY, count

def main():
    tX, tY, numFeatures = getTrainableData()

    print ("Processed", numFeatures, "entries from the training data.")
    # andrei: u can see the contents of the generated list here
    for tx, ty in zip(tX, tY):
        print (ty, tx)



if __name__ == "__main__":
    main()










