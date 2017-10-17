import featureExtractor as fe
import numpy as np
import os
from os.path import isfile, join

outputFilename = "result_merged"
folder = "data/outputs/"
predictCount = 118508

def estimateWeights(ps, Epsilon=0):
    d2p = 1.0 + Epsilon - ps
    weights = ps / d2p
    return weights / weights.sum() # normalized

def mergeResults():
    filenames = [f for f in os.listdir(folder) if isfile(join(folder, f)) and f.__contains__(".csv")]

    print(filenames)

    weights = np.zeros(len(filenames))
    for index, filename in enumerate(filenames):
        with open(folder + filename, "r", encoding="utf8") as file:
            weights[index] = float(next(file))
    weights = estimateWeights(weights)

    probabilities = np.zeros((predictCount, len(fe.languageNames)))
    for index, filename in enumerate(filenames):
        with open(folder + filename, "r", encoding="utf8") as file:
            print("reading file", filename)

            # skip header line
            next(file)

            count = 0
            while True:
                try:
                    label = fe.processYLine(next(file))
                    probabilities[count, label] += weights[index]
                    count += 1
                except StopIteration:
                    break

    with open("{}-{}.csv".format(outputFilename, len(filenames)), "w") as outputFile:
        outputFile.write("Id,Category\n")
        for i in range(predictCount):
            predict = probabilities[i].argmax()
            line = "{},{}".format(i, predict)
            outputFile.write("{}\n".format(line))

def compare(filename1, filename2):
    with open(filename1, "r", encoding="utf8") as f1, open(filename2, "r", encoding="utf8") as f2:
        # skip header
        next(f1)
        next(f2)
        total = 0
        mistmatch = 0
        for line1, line2 in zip(f1, f2):
            np.zeros(len(fe.languageNames))
            y1 = fe.processYLine(line1)
            y2 = fe.processYLine(line2)
            total += 1
            if not y1 == y2:
                mistmatch += 1
        print(mistmatch, total, "ratio:", mistmatch / total)

def main():
    mergeResults()
    # compare("{}-7.csv".format(outputFilename), folder + "test_result_7.csv")

if __name__=="__main__":
    main()
