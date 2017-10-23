from random import random

import math

import featureExtractor as fe

BEST_FEATURES_FORCE_RECOMPUTE=True
FULL_DICT_FORCE_RECOMPUTE=False

samplesToGenerate = 200000
folder = "data/"
outputFilenameX = folder + "generatedTestSetX-{}.csv".format(samplesToGenerate)
outputFilenameY = folder + "generatedTestSetY-{}.csv".format(samplesToGenerate)

minSentenceLength = 4
maxSentenceLength = 30

def getRandomizedSample(ngramDict, ngramLangCounts):
    length = int(minSentenceLength + (maxSentenceLength - minSentenceLength) * random())

    sumLangCounts = ngramLangCounts.sum()
    langs = sorted(enumerate(ngramLangCounts), key=lambda pair: pair[1], reverse=True)

    rand = random() * sumLangCounts
    count = 0
    lang = 0
    for _lang, _count in langs:
        count += _count

        if count >= rand:
            lang = _lang
            break

    sentence = ""
    for _ in range(length):
        stop = ngramLangCounts[lang] * random()

        count = 0
        for ngram, (counts, index) in ngramDict.items():
            count += counts[lang]
            if count >= stop:
                sentence += ngram + " "
                break

    return lang, sentence

def main():
    ngramsList, ngramDict, ngramLangCounts = fe.processDictionaryAsTrainingFeatures(
        force_recompute_best=BEST_FEATURES_FORCE_RECOMPUTE, force_recompute_dict=FULL_DICT_FORCE_RECOMPUTE)

    totalCount = ngramLangCounts.sum()

    with open(outputFilenameX, "w", encoding="utf8") as fX, open(outputFilenameY, "w", encoding="utf8") as fY:
        # write headers
        fX.write("Id,Text\n")
        fY.write("Id,Category\n")

        for i in range(samplesToGenerate):
            lang, sentence = getRandomizedSample(ngramDict, ngramLangCounts)
            fX.write("{},{}\n".format(i, sentence))
            fY.write("{},{}\n".format(i, lang))

            if i % 1000 == 0:
                print("\rFinished {}/{}...".format(i, samplesToGenerate), end="")


if __name__=="__main__":
    main()
