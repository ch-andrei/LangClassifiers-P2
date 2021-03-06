import pickle as pk
import queue

import numpy as np

import featureExtractor as fe
import classifier_naiveBayes as nb

# sklearn's classifier implementations
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

########################################################################################################################

# SELECT THE CLASSIFIER TYPE HERE

CLF_type = 'NB'
# select from:
# 'NB'      our own implementation of Naive Bayes
# 'RF'      sklearn's Random Forests Classifier
# 'AB-RF'   sklearn's Random Forests Classifier with Adaboost
# 'AB-SGD'  sklearn's Stochastic Gradient Descent Classifier with Adaboost
# 'NBM'     sklearn's Multinomial Naive Bayes
# 'MLP'     sklearn's Multi-layer Perceptron
# other options will not be run

# Warning, 'AB-RF' takes a LOT of RAM (~10-15GB)

# toggles for validation on fake test set, and prediction on the actual test set
DO_VALIDATE = True
DO_PREDICT = True

########################################################################################################################

# will train even if a .pkl is available
CLASSIFIER_FORCE_RECOMPUTE = True

# recompute the dictionary of ngrams
FULL_DICT_FORCE_RECOMPUTE = False

# recompute the best features from the dictionary (for vectorization)
BEST_FEATURES_FORCE_RECOMPUTE = CLASSIFIER_FORCE_RECOMPUTE or FULL_DICT_FORCE_RECOMPUTE or True

########################################################################################################################
# work in batches in order to minimize RAM usage
# no need to store the entire database at the same time, work only on a chunk of it

# training
trainTotalSize = fe.getTrainXFileLineCount()
trainBatchSize = 1024 * 64
trainBatchCount = int(np.ceil(trainTotalSize / trainBatchSize))

# validation
trainValidationSize = fe.getTrainXFileLineCount(True)
trainValidationBatchCount = int(np.ceil(trainValidationSize / trainBatchSize))

# predicts
predictTotalSize = fe.getTestXFileLineCount()
predictBatchSize = 1024 * 32
predictBatchCount = int(np.ceil(predictTotalSize / predictBatchSize))

########################################################################################################################
# Adaboost classifier configuration

if CLF_type == 'AB-RF':
    AB_n_estimators_max = 64
else:
    AB_n_estimators_max = 1024

AB_n_estimators_per_batch = min(AB_n_estimators_max, max(1, int(AB_n_estimators_max / trainBatchCount)))
AB_learning_rate = 0.05

AB_RF_n_estimators = 64
AB_RF_max_depth = 512

AB_name = "{}_{}-{}grams_{}n-est_{}RF-n-est_{}learnRate_{}train-size".format(
    CLF_type, fe.ngramMin, fe.ngramMax, AB_RF_n_estimators,
    AB_n_estimators_max, AB_learning_rate, trainTotalSize)

########################################################################################################################
# Random-Forest classifier configuration

RF_n_estimators_max = 256
RF_n_estimators_per_batch = min(RF_n_estimators_max, max(1, int(RF_n_estimators_max / trainBatchCount)))
RF_max_depth = 1024
RF_name = "RF_{}-{}grams_{}n-est_{}max-dep_{}train-size".format(fe.ngramMin, fe.ngramMax, RF_n_estimators_max,
                                                                RF_max_depth, trainTotalSize)

########################################################################################################################

NBM_alpha = 0.1
NBM_name = "NBM_alpha{}_{}-{}grams_{}train-size".format(NBM_alpha, fe.ngramMin, fe.ngramMax, trainTotalSize)

########################################################################################################################

MLP_name = "MLP_{}-{}grams_{}train-size".format(fe.ngramMin, fe.ngramMax, trainTotalSize)

########################################################################################################################

NB_name = "NB_{}-{}grams_{}train-size".format(fe.ngramMin, fe.ngramMax, trainTotalSize)

########################################################################################################################

if CLF_type == 'RF':
    print("Using RandomForests classifier.")
    CLF_name = RF_name
elif CLF_type == 'AB-RF':
    print("Using Adaboost classifier with RandomForests.")
    CLF_name = AB_name
elif CLF_type == 'AB-SGD':
    print("Using Adaboost classifier with SGD.")
    CLF_name = AB_name
elif CLF_type == 'NBM':
    print("Using Multinomial Naive Bayes classifier.")
    CLF_name = NBM_name
elif CLF_type == 'NB':
    print("Using custom Naive Bayes classifier.")
    CLF_name = NB_name
elif CLF_type == 'MLP':
    print("Using MLP classifier.")
    CLF_name = MLP_name
else:
    print("Unknown classifier selected. Exiting...")
    exit(1)

CLF_PKL_name = CLF_name + '.pkl'
########################################################################################################################

def main():
    print("Using classifier ", CLF_name)
    print("Training classifier with batch ~ sample counts [batch size{}; totals: train {} - validation {}]".format(
        trainBatchSize, trainTotalSize, trainValidationSize))

    ngramsList, ngramDict, ngramLangCounts = fe.processDictionaryAsTrainingFeatures(
        force_recompute_best=BEST_FEATURES_FORCE_RECOMPUTE, force_recompute_dict=FULL_DICT_FORCE_RECOMPUTE)

    print("Vectorizing based on a dictionary with {} features".format(len(ngramsList)))

    clf = None
    if CLF_type == 'NB':
        # NB does not need to be trained, all the heavy lifting is done when the dictionary is generated
        clf = nb.NaiveBayesClassifier(ngramsList, ngramDict, ngramLangCounts)
    else:
        if not CLASSIFIER_FORCE_RECOMPUTE:
            clf = fe.pickleReadOrWriteObject(filename=CLF_PKL_name)
        elif clf == None:
            print("Recomputing classifier...")
            clfs = queue.Queue()
            batchNum = 0
            while batchNum < trainBatchCount:
                rawtX, tY = fe.readRawTrainingLines(trainTotalSize, trainBatchSize, batchNum * trainBatchSize)
                tX = fe.vectorizeLines(rawtX, ngramDict)

                if len(rawtX) == 0:
                    break

                tX = np.array(tX)

                if CLF_type == 'MLP':
                    # ~ 0.77
                    clf = MLPClassifier(solver='adam',
                                        alpha=1e-4,
                                        hidden_layer_sizes=(142,140,70,68,34,32,16),
                                        activation='relu',
                                        max_iter=75,
                                        shuffle=True,
                                        warm_start=True,
                                        early_stopping=True,
                                        validation_fraction=0.15
                                        )

                elif CLF_type == 'NBM':
                    # ~0.77
                    clf = MultinomialNB(alpha=NBM_alpha)
                elif CLF_type == 'RF':
                    # ~0.75583
                    clf = RandomForestClassifier(n_estimators=RF_n_estimators_per_batch,
                                                 max_depth=RF_max_depth,
                                                 n_jobs=fe.numProcesses,
                                                 bootstrap=True
                                                 )
                elif CLF_type == 'AB-RF':
                    # ~0.75766
                    clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=AB_RF_n_estimators,
                                                                                   max_depth=AB_RF_max_depth,
                                                                                   n_jobs=fe.numProcesses,
                                                                                   bootstrap=True
                                                                                   ),
                                             n_estimators=AB_n_estimators_per_batch,
                                             learning_rate=AB_learning_rate
                                             )
                else:
                    # 0.73905
                    clf = AdaBoostClassifier(base_estimator=SGDClassifier(loss='perceptron',
                                                                          n_jobs=fe.numProcesses,
                                                                          learning_rate='optimal',
                                                                          max_iter=50
                                                                          ),
                                             algorithm="SAMME",
                                             n_estimators=AB_n_estimators_per_batch,
                                             learning_rate=AB_learning_rate
                                            )

                if CLF_type == 'MLP':
                    clf.fit(tX, tY)
                else:
                    classWeights = fe.getTrainClassWeights(tY)
                    sampleWeights = np.array([classWeights.sum() / classWeights[tY[i]] for i in range(len(tY))])

                    if CLF_type == 'NBM':
                        classes = [i for i in fe.languageNames]
                        clf.partial_fit(tX, tY, classes=classes)
                    else:
                        clf.fit(tX, tY)

                if not CLF_type == 'NBM':
                    clfs.put(clf)

                batchNum += 1
                print("\rTraining classifiers progress {}/{}...".format(batchNum, trainBatchCount), end="")

            print('')

            if not CLF_type == 'NBM' and not CLF_type == 'MLP':
                if clfs.empty():
                    print("Empty classifier queue; nothing was generated.")
                    exit(1)

                # merge clfs from queue
                clf = clfs.get()
                while not clfs.empty():
                    _clf = clfs.get()
                    clf.estimators_.extend(_clf.estimators_)
                    clf.n_estimators += _clf.n_estimators

            # AB-RF uses too much memory
            if not CLF_type == 'AB-RF':
                # write RF to pickle
                fe.pickleReadOrWriteObject(filename=CLF_PKL_name, object=clf)

            print("Finished training on {} samples.".format(trainTotalSize))

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
    else:
        print("Not running validation.")

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

        outputFilename = "{}.csv".format(CLF_name)
        with open(outputFilename, "w") as outputFile:
            for line in predicts:
                outputFile.write("{}\n".format(line))
        print("Wrote predicts to", outputFilename)
    else:
        print("Not running predicts.")

if __name__ == "__main__":
    main()
