import numpy as np

class NaiveBayesClassifier:
    def __init__(self, ngramsList, ngramDict, ngramLangCounts):
        self.ngramsList = ngramsList
        self.ngramDict = ngramDict
        self.ngramLangCounts = ngramLangCounts
        self.ngramLangCountsSum = ngramLangCounts.sum()
        self.pLang = ngramLangCounts / self.ngramLangCountsSum

    def pNgram(self, i):
        ngram = self.ngramsList[i]
        pN = self.ngramDict[ngram][0] / self.ngramDict[ngram][0].sum()
        pL = self.pLang # self.ngramLangCounts / self.ngramLangCountsSum
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
