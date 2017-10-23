import math
import time
from collections import Counter
from random import shuffle
import pickle
import pandas as pd

# split train and validation sets
def split_data(data, VALIDATION_FRAC):
    # random shuffle then split training, validation sets
    shuffle(data)
    split = int(VALIDATION_FRAC*len(data))
    validation = data[:split]
    train = data[split:]
    return train, validation

# save object to pickle file
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# get mean and std deviation of each feature upon each class
def generate_stats(tr, TOP_N_FEATURE):
    t0 = time.time()
    # stats : {class1: {feature1: P1, feature2: P2, ...}, class2: {...},...}
    stats = {}
    # stores sample in class1 / total sample
    p_cls = {}
    # find unique class list
    classes = list(set([x[1] for x in tr]))
    # initial data structures
    top_features = {cls: Counter() for cls in classes}
    total_freq_per_class = {cls: 0 for cls in classes}
    features = []
    # rank features in classes based on frequency
    for sample in tr:
        top_features[sample[1]] += Counter(sample[0])
        total_freq_per_class[sample[1]] += sum(sample[0].values())

    # output for Counter.most_common(2): [('w', 28), ('r', 24)]
    top_features = [x.most_common(TOP_N_FEATURE) for x in top_features.values()]
    for x in top_features:
        features += [y[0] for y in x]
    # get unique feature list
    features = list(set(features))
    nfeature = len(features)
    print ('# features: ',nfeature)

    # generate Naive Bayes Model, store in stats
    for cls in classes:
        stats[cls] = {}
        docs_in_cls = [x[0] for x in tr if x[1]==cls]
        p_cls[cls] = float(len(docs_in_cls))/len(tr)
        for feature in features:
            feature_in_cls = 0
            for sample in docs_in_cls:
                if feature in sample.keys():
                    feature_in_cls += sample[feature]
            stats[cls][feature] = float(1 + feature_in_cls)/(nfeature + total_freq_per_class[cls])

    # save model
    NUM_SAMPLES = len(tr)
    print('Saving Model to {}'.format( 'model_{}Train_{}Feature.plk'.format(NUM_SAMPLES,nfeature)))
    save_obj([stats,p_cls], 'model_{}Train_{}Feature'.format(NUM_SAMPLES,nfeature))

    return stats, p_cls

# return probability for each feature
def featureProbability(cls, feature, freq, stats):
    if feature in stats[cls].keys():
        return math.pow(stats[cls][feature], freq)/math.factorial(freq)
    else:
        return 1

# get probabilities of current sample belongs to each class
def calculateClassProbability(sample, stats, p_cls):
    cls_probabilities = {}
    for cls, feature_stats in stats.iteritems():
        cls_probabilities[cls] = p_cls[cls]
        for feature, freq in sample.iteritems():
            cls_probabilities[cls] *= featureProbability(cls, feature, freq, stats)
            #cls_probabilities[cls] += math.log(featureProbability(cls, feature, freq, stats),10)
    return cls_probabilities

# return the class with the maximum probability as predition value
def predict(sample, stats, p_cls):
    probabilities = calculateClassProbability(sample, stats, p_cls)
    #print(probabilities)
    best_cls = max(probabilities, key=probabilities.get)
    return best_cls

# get predition value for each sample
def run_test(tests, stats, p_cls):
    predictions = []
    start_time = time.time()
    for sample in tests:
        predictions.append(predict(sample, stats, p_cls))
    print('Testing time: ', time.time() - start_time)
    return  predictions

# get accuracy for validation set
def get_accuracy(answer, predictions):
    num_correct = 0
    for x,y in zip(answer, predictions):
        if x==y:
            num_correct += 1
    print('Accuracy: ', float(num_correct) / len(answer) * 100, '%')


# run training, validation, and testing
def run_prediction(data, test, VALIDATION_FRAC, TOP_N_FEATURE, TRIAL, load_model=False, modelFile=None, dataFolderName=None,testResultFile=None):
    # split train and validation set
    train, validation = split_data(data, VALIDATION_FRAC)
    print('Dataset split into training {}, validation {}'.format(len(train), len(validation)))

    # run training/ or loading existing model
    start_time = time.time()
    if load_model==False:
        print('_______________TRAINING________________')
        # generate model
        stats, p_cls = generate_stats(train, TOP_N_FEATURE)
    else:
        print('_____________LOADING_MODEL______________')
        # load model
        if modelFile==None:
            print('Please specify model.pkl you want to load ...')
            exit()
        #model_name = 'C:\Users\guanqing\OneDrive\School\FALL 2017\COMP 551\Assignment2\LangClassifiers-P2\Naive Bayes\model_1-1gram_10000Train_128Feature.pkl'
        model = pickle.load(open(modelFile,'rb'))
        stats = model[0]
        p_cls = model[1]
    print('Traing time: ', time.time()-start_time)

    # run validation
    print('______________VALIDATION_______________')
    validation_sample = [x[0] for x in validation]
    validation_answer = [x[1] for x in validation]
    v_predictions = run_test(validation_sample, stats, p_cls)
    get_accuracy(validation_answer, v_predictions)

    # run testing
    print('________________TESTING_________________')
    t_predictions = run_test(test, stats, p_cls)
    outfile = 'Id,Category\n'
    for i in range(len(t_predictions)):
        outfile += '{},{}\n'.format(i, t_predictions[i])
    f = open('test_result_{}.csv'.format(TRIAL), 'w')
    f.write(outfile)
    f.close()


    # if test reault is provided -> get test accuracy
    if testResultFile!=None:
        dft = pd.read_csv('test_result_{}.csv'.format(TRIAL))
        dfa = pd.read_csv(dataFolderName+testResultFile)

        dfr = dft.merge(dfa, on='Id', how='left')
        #print(dfr.head())
        dfr['result'] = dfr.Category_x == dfr.Category_y
        print('Test Accuracy: ',float(len(dfr[dfr.result == True])) / len(dfr), len(dfr[dfr.result == True]))



