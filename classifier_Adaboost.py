import pickle as pk
import queue

import numpy as np

# sklearn's random forest implementation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

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

AB_FORCE_RECOMPUTE = True

BEST_FEATURES_FORCE_RECOMPUTE = AB_FORCE_RECOMPUTE or True
FULL_DICT_FORCE_RECOMPUTE = False # setting this to True crashes the program for some reason TODO: fix crash

AB_classifier = 'RF' # 'RF' or 'SGD'

########################################################################################################################
# work in batches because this way RAM usage is a lot lower
# means that we dont store the entire database at the same time, but work only with a chunk of it

# training
trainTotalSize = fe.getTrainXFileLineCount()
trainBatchSize = 1024 * 32
trainBatchCount = int(np.ceil(trainTotalSize / trainBatchSize))

DO_VALIDATE = True

trainValidationSize = fe.getTrainXFileLineCount(True)
trainValidationBatchCount = int(np.ceil(trainValidationSize / trainBatchSize))

# predicting
DO_PREDICT = True

predictTotalSize = fe.getTestXFileLineCount()
predictBatchSize = 1024 * 16
predictBatchCount = int(np.ceil(predictTotalSize / predictBatchSize))

print("Training with batch sample counts [train {} - validation {}], batch size {}".format(
    trainBatchCount * trainBatchSize, trainValidationBatchCount * trainBatchSize, trainBatchSize))

########################################################################################################################
if AB_classifier == 'RF':
    AB_n_estimators_max = 16
else:
    AB_n_estimators_max = 512
AB_n_estimators_per_batch = min(AB_n_estimators_max, max(1, int(AB_n_estimators_max / trainBatchCount)))
AB_learning_rate = 0.01

RF_n_estimators = 128

AB_name = "AB-{}_{}-{}grams_{}n-est_{}RF-n-est_{}learnRate_{}train-size".format(
    AB_classifier, fe.ngramMin, fe.ngramMax, RF_n_estimators,
    AB_n_estimators_max, AB_learning_rate, trainTotalSize)

print("Using RandomForest with name", AB_name, "and parameters", AB_n_estimators_max, AB_n_estimators_per_batch, AB_learning_rate)
AB_pickle_name = AB_name + ".pkl"

########################################################################################################################

def main():
    clf = fe.pickleReadOrWriteObject(filename=AB_pickle_name)

    ngramsList, ngramDict, ngramLangCounts = fe.processDictionaryAsTrainingFeatures(
        force_recompute_best=BEST_FEATURES_FORCE_RECOMPUTE, force_recompute_dict=FULL_DICT_FORCE_RECOMPUTE)

    print("Vectorizing dictionary with {} best features".format(len(ngramsList)))

    if AB_classifier == 'RF':
        print("Using Adaboost with RandomForests.")
    else:
        print("Using Adaboost with SGD.")

    if clf == None or AB_FORCE_RECOMPUTE:
        print("Recomputing AB...")
        clfs = queue.Queue()
        batchNum = 0
        while batchNum < trainBatchCount:
            rawtX, tY = fe.readRawTrainingLines(trainTotalSize, trainBatchSize, batchNum * trainBatchSize)
            tX = fe.vectorizeLines(rawtX, ngramDict)

            if len(rawtX) == 0:
                break

            tX = np.array(tX)

            if AB_classifier == 'RF':
                # ~0.75766
                clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=RF_n_estimators,
                                                                               max_depth=512,
                                                                               n_jobs=numProcesses,
                                                                               bootstrap=True
                                                                               ),
                                         n_estimators=AB_n_estimators_per_batch,
                                         learning_rate=AB_learning_rate
                                         )
            else:
                # 0.73905
                clf = AdaBoostClassifier(base_estimator= SGDClassifier(loss='perceptron',
                                                                       n_jobs=numProcesses,
                                                                       learning_rate='optimal'
                                                                       ),
                                         algorithm="SAMME",
                                         n_estimators=AB_n_estimators_per_batch,
                                         learning_rate=AB_learning_rate
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
        fe.pickleReadOrWriteObject(filename=AB_pickle_name, object=clf)

        print("Finished training on {} samples".format(batchNum * trainBatchSize))

    if DO_VALIDATE:
         print("Validation...")
         validationBatchNum = 0
         correct = 0
         total = 0
         while validationBatchNum < trainValidationBatchCount:
             rawtX, tY = fe.readRawTrainingLines(trainValidationSize, trainBatchSize,
                                                 (validationBatchNum) * trainBatchSize, True)

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

    if DO_PREDICT:
        print("Prediction...")

        predicts = []
        # put header to predicts list
        predicts.append("Id,Category")

        batchNum = 0
        while batchNum < predictBatchSize:
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

        outputFilename = "{}.csv".format(AB_name)
        with open(outputFilename, "w") as outputFile:
            for line in predicts:
                outputFile.write("{}\n".format(line))
        print("Wrote predicts to", outputFilename)
    else:
        print("Not running predicts.")

if __name__ == "__main__":
    main()
