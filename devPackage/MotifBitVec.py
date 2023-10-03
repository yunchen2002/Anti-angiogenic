import re
import pandas as pd


def rename(df, name):
    df = df.drop([0], axis=1)
    result = df.loc[df.index[0]].values.tolist()
    for i in range(len(result)):
        result[i] = name + '_' + result[i]
    df = df.drop([0], axis=0)
    df.set_axis(result, axis='columns', inplace=True)
    return df


class MotifBitVec:
    def __init__(self, seqDict, featureDict):
        self.fastaLi = seqDict.items()
        self.seqsNameLi = list(seqDict.keys())
        self.featureDf = pd.DataFrame()
        if featureDict["Usage"] is True:
            MotifBitVecDf = pd.DataFrame(self.MotifBitVec(featureDict["motifList"]))
            MotifBitVecDf = rename(MotifBitVecDf, "MotifBitVec")
            self.featureDf = pd.concat([self.featureDf, MotifBitVecDf], axis=1)

    def MotifBitVec(self, MotifList):
        encodings = []
        header = ['#']
        for key in MotifList:
            header.append(key)
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            myDict = {}
            for key in MotifList:
                if sequence.count(key) >= 1:
                    myDict[key] = myDict.get(key, 0) + 1
                else:
                    myDict[key] = myDict.get(key, 0) + 0

            for key in MotifList:
                code.append(myDict[key])
            encodings.append(code)
        return encodings

    def getOutputDf(self):
        self.featureDf.index = self.seqsNameLi
        return self.featureDf

