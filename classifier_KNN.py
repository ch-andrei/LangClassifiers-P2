import numpy as np
import featureExtractor as fe

def printVector(vector, ngramList):
    print("Printing vector:")
    for index, count in enumerate(vector):
        if count != 0:
            print(ngramsList[index], count, index)

ngramsList, ngramDict, ngramLangCounts = fe.processDictionaryAsTrainingFeatures()

line = "1,hababa"
vector = fe.vectorizeLine(line, ngramDict)

printVector(vector, ngramsList)

