import numpy as np

# note that all computations are done on numpy arrays and not single values
# thus the probabilities are compute for each language in matrix form (vectorized with numpy)
class NaiveBayesClassifier:
    def __init__(self, ngramsList, ngramDict, ngramLangCounts):
        self.ngramsList = ngramsList
        self.ngramDict = ngramDict
        self.ngramLangCounts = ngramLangCounts
        self.ngramLangCountsSum = ngramLangCounts.sum()
        self.pLang = ngramLangCounts / self.ngramLangCountsSum

    # probability of a given ngram in the ngram dictionary
    def pNgram(self, i):
        ngram = self.ngramsList[i]
        pN = self.ngramDict[ngram][0] / self.ngramDict[ngram][0].sum()
        pL = self.pLang # self.ngramLangCounts / self.ngramLangCountsSum # constant, already computed in constructor
        return pN / pL

    def pVector(self, vector):
        p = np.copy(self.pLang)
        for i in np.where(vector > 0)[0]:
            p *= np.power(self.pNgram(i), int(vector[i]))
        return p

    # same function name as sklearn's classifier
    def predict(self, x):
        if len(x.shape) == 1:
            return np.argmax(self.pVector(x))
        else:
            y = []
            for i in range(x.shape[0]):
                p = self.pVector(x[i])
                y.append(np.argmax(p))
            return y
