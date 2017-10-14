import pandas as pd
import math
import time

def split_df(df, frac):
    # random shuffle then split training, validation sets
    df = df.sample(frac=1).reset_index(drop=True)
    split = int(VALIDATION_FRAC*len(df))
    vdf = df[:split]
    tdf = df[split:]
    print vdf.shape, tdf.shape
    return vdf, tdf

# get mean and std deviation of each feature upon each class
def generate_stats(df):
    stats = {}
    for i in df[CLASS_COL].unique():
        #stats[i] = {}
        stats[i] = []
    for cls, group in df.groupby(CLASS_COL):
        for feature in group.columns[:-1]:
            #stats[cls][feature] = [group[feature].mean(), group[feature].std()]
            stats[cls].append([group[feature].mean(), group[feature].std()])
    return stats


# use Gaussian function to estimate the probability of a given attribute value
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

# make prediction: each feature has a prob, take the product of them
#test1 = {0:3.0, 1:129.0, 2:64.0, 3:29.0, 4:115.0, 5:26.4, 6: 0.219, 7:28.0}
test = [6.0 , 103.0 , 72.0 , 32.0,  190.0,  37.7,  0.324,  55.0]
def calculateClassProbability(test, stats):
    probabilities = {}
    for cls, feature_stats in stats.iteritems():
        probabilities[cls] = 1
        '''for feature in feature_stats.keys():
            mean, stdev = feature_stats[feature]
            x = test[feature]
            probabilities[cls] *= calculateProbability(x, mean, stdev)
        '''
        for feature_index in range(len(feature_stats)):
            mean, stdev = feature_stats[feature_index]
            x = test[feature_index]
            probabilities[cls] *= calculateProbability(x, mean, stdev)
    return probabilities

# predict
def predict(test, stats):
    probabilities = calculateClassProbability(test, stats)
    best_cls = max(probabilities, key=probabilities.get)
    correct_flag = (best_cls == test[-1])
    return best_cls, correct_flag

def run_prediction(df):
    vdf, tdf = split_df(df, VALIDATION_FRAC)
    print 'Dataset split into training {}, validation {}'.format(len(tdf), len(vdf))
    start_time = time.time()
    stats = generate_stats(tdf)
    print 'Traing time: ', time.time()-start_time
    start_time = time.time()
    num_correct = 0
    for row in vdf.iterrows():
        index, data = row
        best_cls, correct = predict(data.tolist(), stats)
        if correct:
            num_correct += 1
    print 'Testing time: ', time.time()-start_time
    print 'Accuracy: ', float(num_correct)/len(vdf)*100, '%'


if __name__ == '__main__':
    VALIDATION_FRAC = 0.3
    in_file = 'pima-indians-diabetes.data.csv'
    CLASS_COL = 8
    NUM_CLASS = 2

    # read data
    df = pd.read_csv(in_file, dtype=float, header=None)

    run_prediction(df)

