import pickle as pk
import queue

import numpy as np

# sklearn's random forest implementation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

import featureExtractor as fe

########################################################################################################################

import multiprocessing as mp

useAllCpus = True
if useAllCpus:
    numProcesses = mp.cpu_count()
else:
    numProcesses = max(1, int(mp.cpu_count() / 2)) # will use only half of the available CPUs

########################################################################################################################

FORCE_RECOMPUTE = True

BEST_FEATURES_FORCE_RECOMPUTE = FORCE_RECOMPUTE or True
FULL_DICT_FORCE_RECOMPUTE = False # setting this to True crashes the program for some reason TODO: fix crash

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

########################################################################################################################
# SELECT CLASSIFIER TYPE HERE

CLF_type = 'AB-SGD'  # select from: ['RF', 'AB-RF', 'AB-SGD']
# Warning, 'AB-RF' takes a LOT of RAM (~12GB) and will most likely lag your computer

########################################################################################################################
# Adaboost classifier configuration

if CLF_type == 'AB-RF':
    AB_n_estimators_max = 16
else:
    AB_n_estimators_max = 512

AB_n_estimators_per_batch = min(AB_n_estimators_max, max(1, int(AB_n_estimators_max / trainBatchCount)))
AB_learning_rate = 0.01

AB_RF_n_estimators = 128
AB_RF_max_depth = 512

AB_name = "{}_{}-{}grams_{}n-est_{}RF-n-est_{}learnRate_{}train-size".format(
    CLF_type, fe.ngramMin, fe.ngramMax, AB_RF_n_estimators,
    AB_n_estimators_max, AB_learning_rate, trainTotalSize)

########################################################################################################################
# Random-Forest classifier configuration

RF_n_estimators_max = 256
RF_n_estimators_per_batch = min(RF_n_estimators_max, max(1, int(RF_n_estimators_max / trainBatchCount)))
RF_max_depth = 512
RF_name = "RF_{}-{}grams_{}n-est_{}max-dep_{}train-size".format(fe.ngramMin, fe.ngramMax, RF_n_estimators_max, RF_max_depth, trainTotalSize)

########################################################################################################################

if CLF_type == 'RF':
    print("Using RandomForests classifier.")
    CLF_name = RF_name
elif CLF_type == 'AB-RF':
    print("Using Adaboost with RandomForests.")
    CLF_name = AB_name
elif CLF_type == 'AB-SGD':
    print("Using Adaboost with SGD.")
    CLF_name = AB_name
else:
    print("Unknown classifier selected. Exiting...")
    exit(1)

CLF_PKL_name = CLF_name + '.pkl'
########################################################################################################################

def main():
    print("Using classifier ", CLF_PKL_name)
    print("Training classifier with batch ~ sample counts [train {}/{} - validation {}/{}]".format(
        trainBatchSize, trainTotalSize, trainValidationSize, trainValidationSize))

    clf = fe.pickleReadOrWriteObject(filename=CLF_PKL_name)

    ngramsList, ngramDict, ngramLangCounts = fe.processDictionaryAsTrainingFeatures(
        force_recompute_best=BEST_FEATURES_FORCE_RECOMPUTE, force_recompute_dict=FULL_DICT_FORCE_RECOMPUTE)

    print("Vectorizing based on a dictionary with {} features".format(len(ngramsList)))

    if clf == None or FORCE_RECOMPUTE:
        print("Recomputing classifier...")
        clfs = queue.Queue()
        batchNum = 0
        while batchNum < trainBatchCount:
            rawtX, tY = fe.readRawTrainingLines(trainTotalSize, trainBatchSize, batchNum * trainBatchSize)
            tX = fe.vectorizeLines(rawtX, ngramDict)

            if len(rawtX) == 0:
                break

            tX = np.array(tX)

            if CLF_type == 'RF':
                # ~0.75583
                clf = RandomForestClassifier(n_estimators=RF_n_estimators_per_batch,
                                             max_depth=RF_max_depth,
                                             n_jobs=numProcesses,
                                             bootstrap=True
                                             )
            elif CLF_type == 'AB-RF':
                # ~0.75766
                clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=AB_RF_n_estimators,
                                                                               max_depth=AB_RF_max_depth,
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
        fe.pickleReadOrWriteObject(filename=CLF_PKL_name, object=clf)

        print("Finished training on {} samples".format(trainTotalSize))

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
        while batchNum < predictBatchCount:
            rawtX = fe.readRawTestingLines(predictTotalSize, predictBatchSize, batchNum * predictBatchSize)

            if not len(rawtX) == 0:
                tX = fe.vectorizeLines(rawtX, ngramDict)

                tX = np.array(tX)

                tY = clf.predict(tX)

                for index, predict in enumerate(tY):
                    predictLine = "{},{}".format(batchNum * predictBatchSize + index, predict)
                    predicts.append(predictLine)
            else:
                print("Warning: Empty rawTx during predict at batch {}".format(batchNum))

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
