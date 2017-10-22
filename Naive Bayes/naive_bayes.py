import math
import time
from collections import Counter
from random import shuffle
import pickle
import pandas as pd

import featureExtractor as fe

VALIDATION_FRAC = 0.05
TOP_N_FEATURE = 1000
TRIAL = 10

NUM_SAMPLES = fe.NUM_SAMPLES
MIN_GRAM = fe.MIN_GRAM
MAX_GRAM = fe.MAX_GRAM

def split_data(data):
    # random shuffle then split training, validation sets
    shuffle(data)
    split = int(VALIDATION_FRAC*len(data))
    validation = data[:split]
    train = data[split:]
    return train, validation

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
# get mean and std deviation of each feature upon each class
def generate_stats(tr):
    print ('_______________TRAINING________________')
    # stores {class1: {feature1: P1, feature2: P2, ...}, class2: {...},...}
    stats = {}
    # stores sample in class1 / total sample
    p_cls = {}
    # set() returns unique-value list
    classes = list(set([x[1] for x in tr]))

    manual_setup = True
    # initialize with old results
    total_freq_per_class = {0: 367919, 1: 3467893, 2: 1539078, 3: 914073, 4: 280178}
    # all features of 200k fake trainset, w/o space
    features = ['\x83', '\x87', '\x8b', '\x8f', '\x93', '\x97', '\x9b', '\x9f', '\xa3', '\xa7', '\xab', '\xaf', '0',
                '\xb3', '4', '\xb7', '8', '\xbb', '\xbf', '\xc3', '\xcb', '\xcf', '\xd7', '\xdb', '\xe3', 'd', 'h', 'l',
                '\xef', 'p', 't', 'x', '\x80', '\x84', '\x88', '\x8c', '\x90', '\x94', '\x98', '\x9c', '\xa0', '\xa4',
                '\xa8', '\xac', '\xb0', '3', '\xb4', '7', '\xb8', '\xbc', '\xc4', '\xcc', '\xd0', '\xd4', '\xd8',
                '\xe0', 'c', '\xe4', 'g', 'k', 'o', '\xf0', 's', 'w', '\x7f', '\x81', '\x85', '\x89', '\x8d', '\x91',
                '\x95', '\x99', '\x9d', '\xa1', '\xa5', '\xa9', '\xad', '\xb1', '2', '\xb5', '6', '\xb9', '\xbd',
                '\xc5', '\xcd', '\xd1', '\xd9', '\xe1', 'b', '\xe5', 'f', 'j', 'n', 'r', 'v', 'z', '\x82', '\x86',
                '\x8a', '\x8e', '\x92', '\x96', '\x9a', '\x9e', '\xa2', '\xa6', '\xaa', '\xae', '1', '\xb2', '5',
                '\xb6', '9', '\xba', '\xbe', '\xc2', '\xc6', '\xca', '\xce', 'a', '\xe2', 'e', 'i', 'm', 'q', 'u', 'y']

    # generate featuer counts and feature list if not manually set
    if manual_setup==False:
        top_features = {cls: Counter() for cls in classes}
        total_freq_per_class = {cls: 0 for cls in classes}
        t0 = time.time()
        for sample in tr:
            top_features[sample[1]] += Counter(sample[0])
            total_freq_per_class[sample[1]] += sum(sample[0].values()) #1,2,3grams: {0: 2080596, 1: 19590790, 2: 8545051, 3: 5000797, 4: 1625121}
        print 'total freq per class',total_freq_per_class
        print ('Training t1: ', time.time()-t0)
        # output of most_common: [('w', 28), ('r', 24)]
        top_features = [x.most_common(TOP_N_FEATURE) for x in top_features.values()]
        for x in top_features:
            print ('#feature/class: ', len(x), x)
            features += [y[0] for y in x]
        #print (len(features), features[:5])
        features = list(set(features))



    nfeature = len(features)
    print 'Features: ', features
    print ('# features: ',nfeature)
    t0 = time.time()
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
            #stats[cls][feature] = float(1 + feature_in_cls) / (2 + total_freq_per_class[cls])
    print ('Training t2: ', time.time()-t0)
    save_obj(stats, 'model_{}-{}gram_{}Train_{}Feature_alg2'.format(MIN_GRAM, MAX_GRAM,NUM_SAMPLES,nfeature))
    print p_cls
    return stats, p_cls


# use Gaussian function to estimate the probability of a given attribute value
#def calculateProbability(x, mean, stdev):
    # Gaussian
	#exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	#return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def featureProbability(cls, feature, freq, stats):
    if feature in stats[cls].keys():
        return math.pow(stats[cls][feature], freq)/math.factorial(freq)
    else:
        return 1

# make prediction: each feature has a prob, take the product of them
def calculateClassProbability(sample, stats, p_cls):
    cls_probabilities = {}
    for cls, feature_stats in stats.iteritems():
        cls_probabilities[cls] = p_cls[cls]
        for feature, freq in sample.iteritems():
            cls_probabilities[cls] *= featureProbability(cls, feature, freq, stats)
            #cls_probabilities[cls] += math.log(featureProbability(cls, feature, freq, stats),10)
    return cls_probabilities


def predict(sample, stats, p_cls):
    probabilities = calculateClassProbability(sample, stats, p_cls)
    #print (probabilities)
    best_cls = max(probabilities, key=probabilities.get)
    return best_cls

def run_test(tests, stats, p_cls):
    predictions = []
    start_time = time.time()
    for sample in tests:
        predictions.append(predict(sample, stats, p_cls))
    print ('Testing time: ', time.time() - start_time)
    return  predictions

def get_accuracy(answer, predictions):
    num_correct = 0
    for x,y in zip(answer, predictions):
        if x==y:
            num_correct += 1
    print ('Accuracy: ', float(num_correct) / len(answer) * 100, '%')

def run_prediction(data, test):
    train, validation = split_data(data)
    print ('Dataset split into training {}, validation {}'.format(len(train), len(validation)))
    start_time = time.time()

    load_model = False
    if load_model==False:
        # generate model
        stats, p_cls = generate_stats(train)
    else:
        # load model
        model_name = 'C:\Users\guanqing\OneDrive\School\FALL 2017\COMP 551\Assignment2\LangClassifiers-P2\Naive Bayes\model_1-1gram_264655Train_112Feature_alg2.pkl'
        stats = pickle.load(open(model_name,'rb'))
        p_cls = {0: 0.051155159483956436, 1: 0.5110376830572803, 2: 0.2528331766219627, 3: 0.13402819281970071, 4: 0.05094578801709994}

    print ('Traing time: ', time.time()-start_time)

    # feature test
    #features = stats_112[0].keys()

    #for i in range(len(features)):
        # remove one featuree at each iteration
    #    fi = features[i]
    #    stats = pickle.load(open('C:\Users\guanqing\OneDrive\School\FALL 2017\COMP 551\Assignment2\LangClassifiers-P2\Naive Bayes\model_1-1gram_264655Train_112Feature_alg2.pkl','rb'))
        # remove feature i from model
    #    for stat in stats.values():
    #        stat.pop(fi)
    #    print '#', i, 'Target feature: ', fi
    #    print len(stats[0]), stats[0].keys()
    print ('______________VALIDATION_______________')
    validation_sample = [x[0] for x in validation]
    validation_answer = [x[1] for x in validation]
    v_predictions = run_test(validation_sample, stats, p_cls)
    get_accuracy(validation_answer, v_predictions)

    print ('_________________TEST__________________')
    t_predictions = run_test(test, stats, p_cls)
    outfile = 'Id,Category\n'
    for i in range(len(t_predictions)):
        outfile += '{},{}\n'.format(i, t_predictions[i])
    f = open('test_result_{}.csv'.format(TRIAL), 'w')
    f.write(outfile)
    f.close()

    # get accuracy
    dft = pd.read_csv(
        'C:\Users\guanqing\OneDrive\School\FALL 2017\COMP 551\Assignment2\LangClassifiers-P2\Naive Bayes\\test_result_10.csv')
    dfa = pd.read_csv(
        'C:\Users\guanqing\OneDrive\School\FALL 2017\COMP 551\Assignment2\LangClassifiers-P2\Naive Bayes\data\generatedTestSetY-100000.csv')

    dfr = dft.merge(dfa, on='Id', how='left')
    print dfr.head()
    dfr['result'] = dfr.Category_x == dfr.Category_y
    print float(len(dfr[dfr.result == True])) / len(dfr), len(dfr[dfr.result == True])



