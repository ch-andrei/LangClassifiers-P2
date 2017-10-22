import os
from os.path import isfile, join

import featureExtractor as fe

folder = "data/"
filename1 = folder + "fixedTrainX.csv"
filename2 = folder + "fixedTrainY.csv"

def cleanDataset():
    with fe.getTrainXFile() as inX, \
            fe.getTrainYFile() as inY, \
            open(filename1, "w", encoding="utf8") as fX, \
            open(filename2, "w", encoding="utf8") as fY:

        # write header
        xH = next(inX)
        yH = next(inY)
        fX.write(xH)
        fY.write(yH)

        count = 0
        for line1, line2 in zip(inX, inY):
            x = line1.split(',')[1]
            y = fe.processYLine(line2)

            if len(x) <= 1:
                continue

            fX.write("{},{}".format(count, x))
            fY.write("{},{}\n".format(count, y))

            count += 1

def mergeDataset():
    mergeFolder = "data/"
    filenames = [("data/generatedTestSetX-200000.csv", "data/generatedTestSetY-200000.csv"),
                 ("data/train_set_x.csv", "data/train_set_y.csv")]
    mergeOutputXFilename = mergeFolder + "mergedDatasetX.csv"
    mergeOutputYFilename = mergeFolder + "mergedDatasetY.csv"

    count=0
    with open(mergeOutputXFilename, "w", encoding="utf8") as mX, open(mergeOutputYFilename, "w", encoding="utf8") as mY:

        mX.write("Id,Text\n")
        mY.write("Id,Category\n")

        for xFile, yFile in filenames:
            print("Merging", xFile, yFile)

            with open(xFile, "r", encoding="utf8") as x, open(yFile, "r", encoding="utf8") as y:

                # skip header
                next(x)
                next(y)

                for line1, line2 in zip(x, y):
                    _x = line1.split(',')[1]
                    _y = fe.processYLine(line2)

                    mX.write("{},{}".format(count, _x))
                    mY.write("{},{}\n".format(count, _y))

                    count += 1

            print("Finished", xFile, yFile)

    print("Finished with total count", count)

def main():
    mergeDataset()

if __name__=="__main__":
    main()
