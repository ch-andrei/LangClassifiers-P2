
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

def main():
    cleanDataset()

if __name__=="__main__":
    main()
