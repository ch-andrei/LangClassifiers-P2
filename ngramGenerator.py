defaultMinN = 2

# computes ngrams for a string based on characters of the string
# example: "i am" -> 2-grams: "i ", " a", "am"
def toCharRangedNgramList(word, nmin = defaultMinN, nmax = None):
    if nmax == None:
        nmax = nmin

    if nmin > nmax:
        raise RuntimeError("Attempting to build ngram list (word) with nmin={} higher than nmax={}.".format(nmin, nmax))

    ngrams = []

    for i in range(nmax - nmin + 1):
        n = nmin + i
        ngrams.extend(toCharNgramList(word, n))

    return ngrams

# computes ngrams for a string based on words of the string
# example: "the quick brown fox" -> 2-grams: "the quick", "quick brown", "brown fox"
def toWordRangedNgramList(line, nmin = defaultMinN, nmax = None):
    if nmax == None:
        nmax = nmin

    if nmin > nmax:
        raise RuntimeError("Attempting to build ngram list (word) with nmin={} higher than nmax={}.".format(nmin, nmax))

    ngrams = []

    for i in range(nmax - nmin + 1):
        ngrams.extend(toWordNgramList(line, nmin + i))

    return ngrams

#######################################################

def toWordNgramList(line, n):
    ngrams = []

    words = line.split(' ')
    for i in range(len(words) - n + 1):
        ngram = words[i]
        for j in range(n - 1):
            ngram += " " + words[i + j + 1]
        ngrams.append(ngram)

    return ngrams

def toCharNgramList(word, n):
    ngrams = []

    chars = list(word)
    for i in range(len(chars) - n + 1):
        ngram = ''
        for j in range(n):
            ngram += (chars[i + j])
        ngrams.append(ngram)

    return ngrams

