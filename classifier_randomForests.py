import pickle as pk
import queue

import numpy as np

# sklearn's random forest implementation
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight as cw

import featureExtractor as fe

########################################################################################################################

import multiprocessing as mp

useAllCpus = True
if useAllCpus:
    numProcesses = mp.cpu_count()
else:
    numProcesses = max(1, int(mp.cpu_count() / 2)) # will use only half of the available CPUs

########################################################################################################################

RF_FORCE_RECOMPUTE = True
BEST_FEATURES_FORCE_RECOMPUTE = False
FULL_DICT_FORCE_RECOMPUTE = False

########################################################################################################################
# work in batches because this way RAM usage is a lot lower
# means that we dont store the entire database at the same time, but work only with a chunk of it

# training
trainTotalSize = fe.getTrainXFileLineCount()
trainBatchSize = 1024 * 16
trainBatchCount = int(np.ceil(trainTotalSize / trainBatchSize))

trainValidationRatio = 0.15
trainValidationBatchCount = int(np.ceil(trainBatchCount * trainValidationRatio))
trainBatchCount -= trainValidationBatchCount

# predicting
predictTotalSize = fe.getTestXFileLineCount()
predictBatchSize = 1024 * 16
predictBatchCount = int(np.ceil(predictTotalSize / predictBatchSize))

print("Training with batch sample counts [train {} - validation {}], batch size {}".format(
    trainBatchCount * trainBatchSize, trainValidationBatchCount * trainBatchSize, trainBatchSize))

########################################################################################################################

RF_n_estimators_max = 256
RF_n_estimators_per_batch = min(RF_n_estimators_max, max(1, int(RF_n_estimators_max / trainBatchCount)))
RF_max_depth = 64
RF_name = "RF_{}-{}grams_{}n-est_{}max-dep_{}train-size".format(fe.ngramMin, fe.ngramMax, RF_n_estimators_max, RF_max_depth, trainTotalSize)

print("Using RandomForest with name", RF_name, "and parameters", RF_n_estimators_max, RF_n_estimators_per_batch, RF_max_depth)
RF_pickle_name = RF_name + ".pkl"

########################################################################################################################

def main():
    clf = fe.pickleReadOrWriteObject(filename=RF_pickle_name)

    ngramsList, ngramDict, ngramLangCounts = fe.processDictionaryAsTrainingFeatures(
        force_recompute_best=BEST_FEATURES_FORCE_RECOMPUTE, force_recompute_dict=FULL_DICT_FORCE_RECOMPUTE)

    print("Vectorizing dictionary with {} best features".format(len(ngramsList)))
    for ngram in ngramsList:
        print (ngram)

    if clf == None or RF_FORCE_RECOMPUTE:
        print("Recomputing RF...")
        clfs = queue.Queue()
        batchNum = 0
        while batchNum < trainBatchCount:
            rawtX, tY = fe.readRawTrainingLines(trainTotalSize, trainBatchSize, batchNum * trainBatchSize)
            tX = fe.vectorizeLines(rawtX, ngramDict)

            if len(rawtX) == 0:
                break

            tX = np.array(tX)

            clf = RandomForestClassifier(n_estimators=RF_n_estimators_per_batch,
                                         max_depth=RF_max_depth,
                                         random_state=0,
                                         n_jobs=numProcesses
                                         )

            classWeights = fe.getTrainClassWeights(tY)
            sampleWeights = np.array([classWeights.sum() / classWeights[tY[i]] for i in range(len(tY))])

            clf.fit(tX, tY, sampleWeights)

            clfs.put(clf)

            batchNum += 1
            print("\rTraining classifiers progress {}/{}...".format(batchNum, trainBatchCount), end="")

        print('')

        if clfs.empty():
            print("Empty classifier queue; nothing was generated.")
            exit(1)

        # merge clfs
        clf = clfs.get()
        while not clfs.empty():
            _clf = clfs.get()
            clf.estimators_.extend(_clf.estimators_)
            clf.n_estimators += _clf.n_estimators

        # write RF to pickle
        fe.pickleReadOrWriteObject(filename=RF_pickle_name, object=clf)

        print("Finished training on {} samples".format(batchNum * trainBatchSize))

        print("Validation...")
        validationBatchNum = 0
        correct = 0
        total = 0
        while validationBatchNum < trainValidationBatchCount:
            rawtX, tY = fe.readRawTrainingLines(trainTotalSize, trainBatchSize,
                                                (batchNum + validationBatchNum) * trainBatchSize)

            if len(rawtX) == 0:
                print("read zero rawtx")
                break

            tX = fe.vectorizeLines(rawtX, ngramDict)

            tX = np.array(tX)

            _tY = clf.predict(tX)

            for y, _y in zip(tY, _tY):
                if y == _y:
                    correct += 1
                total += 1

            validationBatchNum += 1
            print("\rValidation progress {}/{}...".format(validationBatchNum, trainValidationBatchCount), end="")

        print("Validation ratio: {}/{}={}".format(correct, total, correct / total))

    predicts = []
    # put header to predicts list
    predicts.append("Id,Category")

    batchNum = 0
    while batchNum < trainBatchCount:
        rawtX = fe.readRawTestingLines(predictTotalSize, predictBatchSize, batchNum * predictBatchSize)

        if len(rawtX) == 0:
            break

        tX = fe.vectorizeLines(rawtX, ngramDict)

        tX = np.array(tX)

        tY = clf.predict(tX)

        for index, predict in enumerate(tY):
            predictLine = "{},{}".format(batchNum * predictBatchSize + index, predict)
            predicts.append(predictLine)

        batchNum += 1
        print("\rPredicting progress {}/{}...".format(batchNum, predictBatchCount), end="")

    print('')

    with open("{}.csv".format(RF_name), "w") as outputFile:
        for line in predicts:
            outputFile.write("{}\n".format(line))

if __name__ == "__main__":
    main()
