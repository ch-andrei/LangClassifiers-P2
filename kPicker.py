
from sklearn.model_selection import train_test_split
import numpy as np
#import featureExtractor as fe
import pickle
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
lb = preprocessing.LabelEncoder()


df = pickle.load(open( "J:\AML-P2\data\dict_1-3grams_276516Train.pkl", "rb" ))
    
class_ngrams = dict()
def classify_ngrams(data):
    for key in data:
        class_max = np.argmax(data[key])
        class_ngrams[key] = class_max
                    
classify_ngrams(df)

#splitting the training set       
X = np.array(list(class_ngrams.keys()))
y = np.array(list(class_ngrams.values()))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)



#tuning the parameter
#X_train = X_train.reshape(-1, 1)
#X_train_encoded = lb.transform(X_train)
y_train_converted = y_train.reshape(-1, )
X_train_flat = np.ravel(X_train)
X_train_converted = lb.fit_transform(X_train_flat)
X_train_converted = X_train_converted.reshape(-1, 1)
k_range = list(range(1,100))

cv_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X_train_converted, y_train_converted, cv = 10, scoring = 'accuracy')
    cv_scores.append(scores.mean())
    print (cv_scores)
    
# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = k_range[MSE.index(min(MSE))]
print ("The optimal number of neighbors is %d" % optimal_k)


# plot misclassification error vs k
plt.plot(k_range, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()



''' 
def main():
    df = pickle.load(open( "J:\AML-P2\data\dict_1-3grams_276516Train.pkl", "rb" ))
    df.head()

#if __name__ == '__main__':
    main() 
'''