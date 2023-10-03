import re
import pandas as pd
from collections import Counter


def rename(df, name):
    df = df.drop([0], axis=1)
    result = df.loc[df.index[0]].values.tolist()
    for i in range(len(result)):
        result[i] = name + '_' + result[i]
    df = df.drop([0], axis=0)
    df.set_axis(result, axis='columns', inplace=True)
    return df


class OVP:
    def __init__(self, seqDict, featureDict):
        self.fastaLi = seqDict.items()
        self.seqsNameLi = list(seqDict.keys())
        self.featureDf = pd.DataFrame()
        if featureDict["OVPC"] is True:
            OVPCDf = pd.DataFrame(self.OVPC())
            OVPCDf = rename(OVPCDf, "OVPC")
            self.featureDf = pd.concat([self.featureDf, OVPCDf], axis=1)
        if featureDict["OVP"][0] is True:
            OVPDf = pd.DataFrame(self.OVP(featureDict["OVP"][1], featureDict["OVP"][2]))
            OVPDf = rename(OVPDf, "OVP")
            self.featureDf = pd.concat([self.featureDf, OVPDf], axis=1)
        else:
            pass

    def OVPC(self):
        group = {"Aromatic": 'FYWH',
                 "Negative": 'DE',
                 "Positive": 'KHR',
                 "Polar": 'NQSDECTKRHYW',
                 "Hydrophobic": 'AGCTIVLKHFYWM',
                 "Aliphatic": 'IVL',
                 "Tiny": 'ASGC',
                 "Charged": 'KHRDE',
                 "Small": 'PNDTCAGSV',
                 "Imino_acid": 'P'}
        groupKey = group.keys()
        encodings = []
        header = ['#']
        for key in groupKey:
            header.append(key)
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            count = Counter(sequence)
            myDict = {}
            for key in groupKey:
                for aa in group[key]:
                    myDict[key] = myDict.get(key, 0) + count[aa]
            for key in groupKey:
                code.append(myDict[key] / len(sequence))
            encodings.append(code)

        return encodings

    def OVP(self, Nnumber=2, Cnumber=3):
        group = {"Aromatic": 'FYWH',
                 "Negative": 'DE',
                 "Positive": 'KHR',
                 "Polar": 'NQSDECTKRHYW',
                 "Hydrophobic": 'AGCTIVLKHFYWM',
                 "Aliphatic": 'IVL',
                 "Tiny": 'ASGC',
                 "Charged": 'KHRDE',
                 "Small": 'PNDTCAGSV',
                 "Imino_acid": 'P'}
        groupKey = group.keys()
        encodings = []
        header = ['#']

        for CN in range(Nnumber):
            for key in groupKey:
                header.append(key + '_N' + str(CN + 1))
        for CN in range(Cnumber):
            for key in groupKey:
                header.append(key + '_C' + str(CN + 1))
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            Nsequence = sequence[:Nnumber]
            Csequence = sequence[-Cnumber:]

            for myaa in Nsequence:
                for key in groupKey:
                    myDict = {}
                    if myaa in group[key]:
                        myDict[key] = myDict.get(key, 0) + 1
                    else:
                        myDict[key] = myDict.get(key, 0) + 0
                    code.append(myDict[key])
            for myaa in Csequence:
                for key in groupKey:
                    myDict = {}
                    if myaa in group[key]:
                        myDict[key] = myDict.get(key, 0) + 1
                    else:
                        myDict[key] = myDict.get(key, 0) + 0
                    code.append(myDict[key])
            encodings.append(code)
        return encodings

    def getOutputDf(self):
        self.featureDf.index = self.seqsNameLi
        return self.featureDf
