import math
import time
from collections import Counter
from random import shuffle
import pickle
import featureExtractor as fe

VALIDATION_FRAC = 0.05
TOP_N_FEATURE = 100
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
    # get a list of unique features
    #top_features = {cls: Counter() for cls in classes}
    total_freq_per_class = {0: 708148, 1: 6643363, 2: 2897727, 3: 1698315, 4: 551801}#{cls: 0 for cls in classes}#{0: 2080596, 1: 19590790, 2: 8545051, 3: 5000797, 4: 1625121}
    t0 = time.time()
    #for sample in tr:
        #top_features[sample[1]] += Counter(sample[0])
        #total_freq_per_class[sample[1]] += sum(sample[0].values()) #1,2,3grams: {0: 2080596, 1: 19590790, 2: 8545051, 3: 5000797, 4: 1625121}
    print total_freq_per_class
    print ('Training t1: ', time.time()-t0)
    # output of most_common: [('w', 28), ('r', 24)]
    '''top_features = [x.most_common(TOP_N_FEATURE) for x in top_features.values()]
    features = []
    for x in top_features:
        print ('#feature/class: ', len(x), x)
        features += [y[0] for y in x]
    #print (len(features), features[:5])
    features = list(set(features))
    print features'''
    # 93 features features = ['\x87', '\x93', '\x9b', ' ', '\xab', '0', '\xb3', '4', '8', '\xbb', '\xc3', 'D', 'H', 'L', 'P', 'T', 'X', 'd', 'h', 'l', 'p', 't', 'x', '\x80', '\x84', '\x94', '\x98', '\x9c', '3', '7', '\xbc', 'C', '\xc4', 'G', 'K', 'O', 'S', 'W', 'c', 'g', 'k', 'o', 's', 'w', '\x81', '\x85', '\x99', '\x9d', '\xa1', '\xa9', '\xb1', '2', '6', '\xb9', '\xbd', 'B', '\xc5', 'F', 'J', 'N', 'R', 'V', 'Z', 'b', 'f', 'j', 'n', 'r', 'v', 'z', '\x82', '\x86', '\x9a', '\x9e', '\xa6', '1', '5', '9', '\xba', 'A', '\xc2', 'E', 'I', 'M', 'U', 'Y', 'a', '\xe2', 'e', 'i', 'm', 'u', 'y']
    #features = ['\x83', '\x87', '\x8b', '\x8f', '\x93', '\x97', '\x9b', '\x9f', ' ', '\xa3', '\xa7', '\xab', '\xaf', '0', '\xb3', '4', '\xb7', '8', '\xbb', '\xbf', '\xc3', '\xcb', '\xcf', 'd', 'h', 'l', '\xef', 'p', 't', 'x', '\x80', '\x84', '\x88', '\x8c', '\x90', '\x94', '\x98', '\x9c', '\xa0', '\xa4', '\xa8', '\xac', '\xb0', '3', '\xb4', '7', '\xb8', '\xbc', '\xc4', '\xcc', '\xd0', '\xe0', 'c', 'g', 'k', 'o', '\xf0', 's', 'w', '\x81', '\x85', '\x89', '\x8d', '\x91', '\x95', '\x99', '\x9d', '\xa1', '\xa5', '\xa9', '\xad', '\xb1', '2', '\xb5', '6', '\xb9', '\xbd', '\xc5', '\xcd', 'b', 'f', 'j', 'n', 'r', 'v', 'z', '\x82', '\x86', '\x8a', '\x8e', '\x92', '\x96', '\x9a', '\x9e', '\xa2', '\xa6', '\xaa', '\xae', '1', '\xb2', '5', '\xb6', '9', '\xba', '\xbe', '\xc2', '\xc6', '\xca', 'a', '\xe2', 'e', 'i', 'm', 'q', 'u', 'y']

    # feture with no numbers and space
    #features = ['\x83', '\x87', '\x8b', '\x8f', '\x93', '\x97', '\x9b', '\x9f', '\xa3', '\xa7', '\xab', '\xaf', '\xb3', '\xb7', '\xbb', '\xbf', '\xc3', '\xcb', 'd', 'h', 'l', '\xef', 'p', 't', 'x', '\x80', '\x84', '\x88', '\x8c', '\x90', '\x94', '\x98', '\x9c', '\xa0', '\xa4', '\xa8', '\xac', '\xb0', '\xb4', '\xb8', '\xbc', '\xc4', 'c', 'g', 'k', 'o', '\xf0', 's', 'w', '\x81', '\x85', '\x89', '\x8d', '\x91', '\x95', '\x99', '\x9d', '\xa1', '\xa5', '\xa9', '\xad', '\xb1', '\xb5', '\xb9', '\xbd', '\xc5', '\xcd', 'b', 'f', 'j', 'n', 'r', 'v', 'z', '\x82', '\x86', '\x8a', '\x8e', '\x92', '\x96', '\x9a', '\x9e', '\xa2', '\xa6', '\xaa', '\xae', '\xb2', '\xb6', '\xba', '\xbe', '\xc2', '\xc6', '\xca', '\xce', 'a', '\xe2', 'e', 'i', 'm', 'q', 'u', 'y']
    # feature with numbers
    features = ['0','1','2','3','4','5','6','7','8','9','\x83', '\x87', '\x8b', '\x8f', '\x93', '\x97', '\x9b', '\x9f', '\xa3', '\xa7', '\xab', '\xaf', '\xb3', '\xb7', '\xbb', '\xbf', '\xc3', '\xcb', 'd', 'h', 'l', '\xef', 'p', 't', 'x', '\x80', '\x84', '\x88', '\x8c', '\x90', '\x94', '\x98', '\x9c', '\xa0', '\xa4', '\xa8', '\xac', '\xb0', '\xb4', '\xb8', '\xbc', '\xc4', 'c', 'g', 'k', 'o', '\xf0', 's', 'w', '\x81', '\x85', '\x89', '\x8d', '\x91', '\x95', '\x99', '\x9d', '\xa1', '\xa5', '\xa9', '\xad', '\xb1', '\xb5', '\xb9', '\xbd', '\xc5', '\xcd', 'b', 'f', 'j', 'n', 'r', 'v', 'z', '\x82', '\x86', '\x8a', '\x8e', '\x92', '\x96', '\x9a', '\x9e', '\xa2', '\xa6', '\xaa', '\xae', '\xb2', '\xb6', '\xba', '\xbe', '\xc2', '\xc6', '\xca', '\xce', 'a', '\xe2', 'e', 'i', 'm', 'q', 'u', 'y']
    nfeature = len(features)
    #print features
    print ('number of features: ',nfeature)
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
            #stats[cls][feature] = float(1 + feature_in_cls)/(nfeature + total_freq_per_class[cls])
            stats[cls][feature] = float(1 + feature_in_cls) / (2 + total_freq_per_class[cls])
    print ('Training t2: ', time.time()-t0)
    save_obj(stats, 'model_{}-{}gram_{}Train_{}Feature_alg2'.format(MIN_GRAM, MAX_GRAM,NUM_SAMPLES,nfeature))
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
#test = {0:3.0, 1:129.0, 2:64.0, 3:29.0, 4:115.0, 5:26.4, 6: 0.219, 7:28.0}
def calculateClassProbability(sample, stats, p_cls):
    cls_probabilities = {}
    for cls, feature_stats in stats.iteritems():
        cls_probabilities[cls] = p_cls[cls]
        for feature, freq in sample.iteritems():
            cls_probabilities[cls] *= featureProbability(cls, feature, freq, stats)
    return cls_probabilities

# predict
def predict(sample, stats, p_cls):
    probabilities = calculateClassProbability(sample, stats, p_cls)
    #print (probabilities)
    best_cls = max(probabilities, key=probabilities.get)
    #print (best_cls, sample[1])
    #correct_flag = (best_cls == sample[1])
    return best_cls #, correct_flag

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
    stats, p_cls = generate_stats(train)
    print ('Traing time: ', time.time()-start_time)

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

'''
def main():
    # read data
    df = pd.read_csv(in_file, dtype=float, header=None)

    run_prediction(df)

if __name__ == '__main__':
    main()
'''

