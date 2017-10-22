import featureExtractor as fe
import numpy as np
import os
from os.path import isfile, join

outputFilenamePrefix = "result_merged"
folder = "data/outputs/"
predictCount = 118508

def estimateWeights(ps, Epsilon=0):
    d2p = 1.0 + Epsilon - ps
    weights = ps / d2p
    return weights / weights.sum() # normalized

def mergeResults(filenames):
    print("Merging result files", filenames)

    weights = np.zeros(len(filenames))
    for index, filename in enumerate(filenames):
        with open(folder + filename, "r", encoding="utf8") as file:
            weights[index] = float(next(file))
    weights = estimateWeights(weights)
    print("File weights are:", weights)

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

    outputFileName = "{}-{}.csv".format(outputFilenamePrefix, len(filenames))
    with open(outputFileName, "w") as outputFile:
        outputFile.write("Id,Category\n")
        for i in range(predictCount):
            predict = probabilities[i].argmax()
            line = "{},{}".format(i, predict)
            outputFile.write("{}\n".format(line))

    return outputFileName

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
        print("difference %:", 100 * mistmatch / total, filename1, "vs", filename2, mistmatch, total)

def compareAll(filenames):
    n = len(filenames)
    p = np.where(np.tril(np.ones((n,n)), -1) > 0)

    for i, j in zip(p[0], p[1]):
        file1 = filenames[i]
        file2 = filenames[j]
        if not file1 == file2:
            print("\n", file1, "vs", file2,)
            compare(folder + file1, folder + file2)

def compareOneToAll(file, filenames):
    for _file in filenames:
        if not file == _file:
            compare(file, folder + _file)

def main():
    filenames = [f for f in os.listdir(folder) if isfile(join(folder, f)) and f.__contains__(".csv")]
    compareAll(filenames)

    outputFileName = mergeResults(filenames)
    compareOneToAll(outputFileName, filenames)

if __name__=="__main__":
    main()
