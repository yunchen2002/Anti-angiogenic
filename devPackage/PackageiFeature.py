from collections import Counter
import os
import math
import platform
import re
import sys
import numpy as np
import pandas as pd


def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1 + '.' + key2] = 0
    return gPair


def Count(seq1, seq2):
    sum = 0
    for aa in seq1:
        sum = sum + seq2.count(aa)
    return sum


def Count1(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence) * 100)
                    break
        if myCount == 0:
            code.append(0)
    return code


def CalculateKSCTriad(sequence, gap, features, AADict):
    res = []
    for g in range(gap + 1):
        myDict = {}
        for f in features:
            myDict[f] = 0

        for i in range(len(sequence)):
            if i + gap + 1 < len(sequence) and i + 2 * gap + 2 < len(sequence):
                fea = AADict[sequence[i]] + '.' + AADict[sequence[i + gap + 1]] + '.' + AADict[
                    sequence[i + 2 * gap + 2]]
                myDict[fea] = myDict[fea] + 1

        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            res.append((myDict[f] - minValue) / maxValue)

    return res


def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)


def rename(df, name):
    df = df.drop([0], axis=1)
    result = df.loc[df.index[0]].values.tolist()
    for i in range(len(result)):
        result[i] = name + '_' + result[i]
    df = df.drop([0], axis=0)
    df.set_axis(result, axis='columns', inplace=True)
    return df


class iFeature:
    def __init__(self, seqDict, featureDict):
        self.fastaLi = seqDict.items()
        self.seqsNameLi = list(seqDict.keys())
        self.featureDf = pd.DataFrame()
        if featureDict["AAC"] is True:
            AACDf = pd.DataFrame(self.AAC())
            AACDf = rename(AACDf, "AAC")
            self.featureDf = pd.concat([self.featureDf, AACDf], axis=1)
        if featureDict["AAINDEX"] is True:
            AAINDEXDf = pd.DataFrame(self.AAINDEX())
            AAINDEXDf = rename(AAINDEXDf, "AAINDEX")
            self.featureDf = pd.concat([self.featureDf, AAINDEXDf], axis=1)
        if featureDict["CKSAAGP"][0] is True:
            CKSAAGPDf = pd.DataFrame(self.CKSAAGP(gap=featureDict["CKSAAGP"][1]))
            CKSAAGPDf = rename(CKSAAGPDf, "CKSAAGP")
            self.featureDf = pd.concat([self.featureDf, CKSAAGPDf], axis=1)
        if featureDict["CTDC"] is True:
            CTDCDf = pd.DataFrame(self.CTDC())
            CTDCDf = rename(CTDCDf, "CTDC")
            self.featureDf = pd.concat([self.featureDf, CTDCDf], axis=1)
        if featureDict["CTDD"] is True:
            CTDDDf = pd.DataFrame(self.CTDD())
            CTDDDf = rename(CTDDDf, "CTDD")
            self.featureDf = pd.concat([self.featureDf, CTDDDf], axis=1)
        if featureDict["CTDT"] is True:
            CTDTDf = pd.DataFrame(self.CTDT())
            CTDTDf = rename(CTDTDf, "CTDT")
            self.featureDf = pd.concat([self.featureDf, CTDTDf], axis=1)
        if featureDict["CTriad"] is True:
            CTriadDf = pd.DataFrame(self.CTriad())
            CTriadDf = rename(CTriadDf, "CTriad")
            self.featureDf = pd.concat([self.featureDf, CTriadDf], axis=1)
        if featureDict["DDE"] is True:
            DDEDf = pd.DataFrame(self.DDE())
            DDEDf = rename(DDEDf, "DDE")
            self.featureDf = pd.concat([self.featureDf, DDEDf], axis=1)
        if featureDict["DPC"] is True:
            DPCDf = pd.DataFrame(self.DPC())
            DPCDf = rename(DPCDf, "DPC")
            self.featureDf = pd.concat([self.featureDf, DPCDf], axis=1)
        if featureDict["GAAC"] is True:
            GAACDf = pd.DataFrame(self.GAAC())
            GAACDf = rename(GAACDf, "GAAC")
            self.featureDf = pd.concat([self.featureDf, GAACDf], axis=1)
        if featureDict["GDPC"] is True:
            GDPCDf = pd.DataFrame(self.GDPC())
            GDPCDf = rename(GDPCDf, "GDPC")
            self.featureDf = pd.concat([self.featureDf, GDPCDf], axis=1)
        if featureDict["GTPC"] is True:
            self.GTPC()
            GTPCDf = pd.DataFrame(self.GTPC())
            GTPCDf = rename(GTPCDf, "GTPC")
            self.featureDf = pd.concat([self.featureDf, GTPCDf], axis=1)
        if featureDict["KSCTriad"][0] is True:
            KSCTriadDf = pd.DataFrame(self.KSCTriad(gap=featureDict["KSCTriad"][1]))
            KSCTriadDf = rename(KSCTriadDf, "KSCTriad")
            self.featureDf = pd.concat([self.featureDf, KSCTriadDf], axis=1)
        if featureDict["QSOrder"][0] is True:
            QSOrderDf = pd.DataFrame(self.QSOrder(nlag=featureDict["QSOrder"][1], w=featureDict["QSOrder"][2]))
            QSOrderDf = rename(QSOrderDf, "QSOrder")
            self.featureDf = pd.concat([self.featureDf, QSOrderDf], axis=1)
        if featureDict["TPC"] is True:
            TPCDf = pd.DataFrame(self.TPC())
            TPCDf = rename(TPCDf, "TPC")
            self.featureDf = pd.concat([self.featureDf, TPCDf], axis=1)
        if featureDict["SOCN"][0] is True:
            SOCNDf = pd.DataFrame(self.SOCNumber(nlag=featureDict["SOCN"][1]))
            SOCNDf = rename(SOCNDf, "SOCN")
            self.featureDf = pd.concat([self.featureDf, SOCNDf], axis=1)
        if featureDict["APAAC"][0] is True:
            APAACDf = pd.DataFrame(self.APAAC(lambdaValue=featureDict["APAAC"][1], w=featureDict["APAAC"][2]))
            APAACDf = rename(APAACDf, "APAAC")
            self.featureDf = pd.concat([self.featureDf, APAACDf], axis=1)
        if featureDict["Geary"][0] is True:
            GearyDf = pd.DataFrame(self.Geary(nlag=featureDict["Geary"][1]))
            GearyDf = rename(GearyDf, "Geary")
            self.featureDf = pd.concat([self.featureDf, GearyDf], axis=1)
        if featureDict["Moran"][0] is True:
            MoranDf = pd.DataFrame(self.Moran(nlag=featureDict["Moran"][1]))
            MoranDf = rename(MoranDf, "Moran")
            self.featureDf = pd.concat([self.featureDf, MoranDf], axis=1)
        if featureDict["NMBroto"][0] is True:
            NMBrotoDf = pd.DataFrame(self.NMBroto(nlag=featureDict["NMBroto"][1]))
            NMBrotoDf = rename(NMBrotoDf, "NMBroto")
            self.featureDf = pd.concat([self.featureDf, NMBrotoDf], axis=1)
        if featureDict["CKSAAP"][0] is True:
            CKSAAPDf = pd.DataFrame(self.CKSAAP(gap=featureDict["CKSAAP"][1]))
            CKSAAPDf = rename(CKSAAPDf, "CKSAAP")
            self.featureDf = pd.concat([self.featureDf, CKSAAPDf], axis=1)
        if featureDict["BINARY"] is True:
            BINARYDf = pd.DataFrame(self.BINARY())
            BINARYDf = rename(BINARYDf, "BINARY")
            self.featureDf = pd.concat([self.featureDf, BINARYDf], axis=1)
        if featureDict["PAAC"][0] is True:
            PAACDf = pd.DataFrame(self.PAAC(lambdaValue=featureDict["PAAC"][1], w=featureDict["PAAC"][2]))
            PAACDf = rename(PAACDf, "PAAC")
            self.featureDf = pd.concat([self.featureDf, PAACDf], axis=1)
        else:
            pass

    def getOutputDf(self):
        self.featureDf.index = self.seqsNameLi
        return self.featureDf

    def AAC(self):
        AA = 'ARNDCQEGHILKMFPSTWYV'
        encodings = []
        header = ['#']
        for i in AA:
            header.append(i)
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            count = Counter(sequence)
            for key in count:
                count[key] = count[key] / len(sequence)
            code = [name]
            for aa in AA:
                code.append(count[aa])
            encodings.append(code)
        return encodings

    def CKSAAGP(self, gap=5):
        group = {
            'alphaticr': 'GAVLMI',
            'aromatic': 'FYW',
            'postivecharger': 'KRH',
            'negativecharger': 'DE',
            'uncharger': 'STCPNQ'
        }

        AA = 'ARNDCQEGHILKMFPSTWYV'

        groupKey = group.keys()

        index = {}
        for key in groupKey:
            for aa in group[key]:
                index[aa] = key

        gPairIndex = []
        for key1 in groupKey:
            for key2 in groupKey:
                gPairIndex.append(key1 + '.' + key2)

        encodings = []
        header = ['#']
        for g in range(gap + 1):
            for p in gPairIndex:
                header.append(p + '.gap' + str(g))
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            for g in range(gap + 1):
                gPair = generateGroupPairs(groupKey)
                sum = 0
                for p1 in range(len(sequence)):
                    p2 = p1 + g + 1
                    if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                        gPair[index[sequence[p1]] + '.' + index[sequence[p2]]] = gPair[
                                                                                     index[sequence[p1]] + '.' + index[
                                                                                         sequence[p2]]] + 1
                        sum = sum + 1

                if sum == 0:
                    for gp in gPairIndex:
                        code.append(0)
                else:
                    for gp in gPairIndex:
                        code.append(gPair[gp] / sum)

            encodings.append(code)

        return encodings

    def CTDC(self):
        group1 = {
            'hydrophobicity_PRAM900101': 'RKEDQN',
            'hydrophobicity_ARGP820101': 'QSTNGDE',
            'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
            'hydrophobicity_PONP930101': 'KPDESNQT',
            'hydrophobicity_CASG920101': 'KDEQPSRNTG',
            'hydrophobicity_ENGD860101': 'RDKENQHYP',
            'hydrophobicity_FASG890101': 'KERSQD',
            'normwaalsvolume': 'GASTPDC',
            'polarity': 'LIFWCMVY',
            'polarizability': 'GASDT',
            'charge': 'KR',
            'secondarystruct': 'EALMQKRH',
            'solventaccess': 'ALFCGIVW'
        }
        group2 = {
            'hydrophobicity_PRAM900101': 'GASTPHY',
            'hydrophobicity_ARGP820101': 'RAHCKMV',
            'hydrophobicity_ZIMJ680101': 'HMCKV',
            'hydrophobicity_PONP930101': 'GRHA',
            'hydrophobicity_CASG920101': 'AHYMLV',
            'hydrophobicity_ENGD860101': 'SGTAW',
            'hydrophobicity_FASG890101': 'NTPG',
            'normwaalsvolume': 'NVEQIL',
            'polarity': 'PATGS',
            'polarizability': 'CPNVEQIL',
            'charge': 'ANCQGHILMFPSTWYV',
            'secondarystruct': 'VIYCWFT',
            'solventaccess': 'RKQEND'
        }
        group3 = {
            'hydrophobicity_PRAM900101': 'CLVIMFW',
            'hydrophobicity_ARGP820101': 'LYPFIW',
            'hydrophobicity_ZIMJ680101': 'LPFYI',
            'hydrophobicity_PONP930101': 'YMFWLCVI',
            'hydrophobicity_CASG920101': 'FIWC',
            'hydrophobicity_ENGD860101': 'CVLIMF',
            'hydrophobicity_FASG890101': 'AYHWVMFLIC',
            'normwaalsvolume': 'MHKFRYW',
            'polarity': 'HQRKNED',
            'polarizability': 'KMHFRYW',
            'charge': 'DE',
            'secondarystruct': 'GNPSD',
            'solventaccess': 'MSPTHY'
        }

        groups = [group1, group2, group3]
        property = ('hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
                    'hydrophobicity_PONP930101', 'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101',
                    'hydrophobicity_FASG890101', 'normwaalsvolume', 'polarity', 'polarizability',
                    'charge', 'secondarystruct', 'solventaccess')

        encodings = []
        header = ['#']
        for p in property:
            for g in range(1, len(groups) + 1):
                header.append(p + '.G' + str(g))
        encodings.append(header)
        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            for p in property:
                c1 = Count(group1[p], sequence) / len(sequence)
                c2 = Count(group2[p], sequence) / len(sequence)
                c3 = 1 - c1 - c2
                code = code + [c1, c2, c3]
            encodings.append(code)
        return encodings

    def CTDD(self):
        group1 = {
            'hydrophobicity_PRAM900101': 'RKEDQN',
            'hydrophobicity_ARGP820101': 'QSTNGDE',
            'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
            'hydrophobicity_PONP930101': 'KPDESNQT',
            'hydrophobicity_CASG920101': 'KDEQPSRNTG',
            'hydrophobicity_ENGD860101': 'RDKENQHYP',
            'hydrophobicity_FASG890101': 'KERSQD',
            'normwaalsvolume': 'GASTPDC',
            'polarity': 'LIFWCMVY',
            'polarizability': 'GASDT',
            'charge': 'KR',
            'secondarystruct': 'EALMQKRH',
            'solventaccess': 'ALFCGIVW'
        }
        group2 = {
            'hydrophobicity_PRAM900101': 'GASTPHY',
            'hydrophobicity_ARGP820101': 'RAHCKMV',
            'hydrophobicity_ZIMJ680101': 'HMCKV',
            'hydrophobicity_PONP930101': 'GRHA',
            'hydrophobicity_CASG920101': 'AHYMLV',
            'hydrophobicity_ENGD860101': 'SGTAW',
            'hydrophobicity_FASG890101': 'NTPG',
            'normwaalsvolume': 'NVEQIL',
            'polarity': 'PATGS',
            'polarizability': 'CPNVEQIL',
            'charge': 'ANCQGHILMFPSTWYV',
            'secondarystruct': 'VIYCWFT',
            'solventaccess': 'RKQEND'
        }
        group3 = {
            'hydrophobicity_PRAM900101': 'CLVIMFW',
            'hydrophobicity_ARGP820101': 'LYPFIW',
            'hydrophobicity_ZIMJ680101': 'LPFYI',
            'hydrophobicity_PONP930101': 'YMFWLCVI',
            'hydrophobicity_CASG920101': 'FIWC',
            'hydrophobicity_ENGD860101': 'CVLIMF',
            'hydrophobicity_FASG890101': 'AYHWVMFLIC',
            'normwaalsvolume': 'MHKFRYW',
            'polarity': 'HQRKNED',
            'polarizability': 'KMHFRYW',
            'charge': 'DE',
            'secondarystruct': 'GNPSD',
            'solventaccess': 'MSPTHY'
        }

        groups = [group1, group2, group3]
        property = (
            'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
            'hydrophobicity_PONP930101',
            'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
            'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

        encodings = []
        header = ['#']
        for p in property:
            for g in ('1', '2', '3'):
                for d in ['0', '25', '50', '75', '100']:
                    header.append(p + '.' + g + '.residue' + d)
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            for p in property:
                code = code + Count1(group1[p], sequence) + Count1(group2[p], sequence) + Count1(group3[p], sequence)
            encodings.append(code)
        return encodings

    def CTDT(self):
        group1 = {
            'hydrophobicity_PRAM900101': 'RKEDQN',
            'hydrophobicity_ARGP820101': 'QSTNGDE',
            'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
            'hydrophobicity_PONP930101': 'KPDESNQT',
            'hydrophobicity_CASG920101': 'KDEQPSRNTG',
            'hydrophobicity_ENGD860101': 'RDKENQHYP',
            'hydrophobicity_FASG890101': 'KERSQD',
            'normwaalsvolume': 'GASTPDC',
            'polarity': 'LIFWCMVY',
            'polarizability': 'GASDT',
            'charge': 'KR',
            'secondarystruct': 'EALMQKRH',
            'solventaccess': 'ALFCGIVW'
        }
        group2 = {
            'hydrophobicity_PRAM900101': 'GASTPHY',
            'hydrophobicity_ARGP820101': 'RAHCKMV',
            'hydrophobicity_ZIMJ680101': 'HMCKV',
            'hydrophobicity_PONP930101': 'GRHA',
            'hydrophobicity_CASG920101': 'AHYMLV',
            'hydrophobicity_ENGD860101': 'SGTAW',
            'hydrophobicity_FASG890101': 'NTPG',
            'normwaalsvolume': 'NVEQIL',
            'polarity': 'PATGS',
            'polarizability': 'CPNVEQIL',
            'charge': 'ANCQGHILMFPSTWYV',
            'secondarystruct': 'VIYCWFT',
            'solventaccess': 'RKQEND'
        }
        group3 = {
            'hydrophobicity_PRAM900101': 'CLVIMFW',
            'hydrophobicity_ARGP820101': 'LYPFIW',
            'hydrophobicity_ZIMJ680101': 'LPFYI',
            'hydrophobicity_PONP930101': 'YMFWLCVI',
            'hydrophobicity_CASG920101': 'FIWC',
            'hydrophobicity_ENGD860101': 'CVLIMF',
            'hydrophobicity_FASG890101': 'AYHWVMFLIC',
            'normwaalsvolume': 'MHKFRYW',
            'polarity': 'HQRKNED',
            'polarizability': 'KMHFRYW',
            'charge': 'DE',
            'secondarystruct': 'GNPSD',
            'solventaccess': 'MSPTHY'
        }

        groups = [group1, group2, group3]
        property = (
            'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
            'hydrophobicity_PONP930101',
            'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
            'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

        encodings = []
        header = ['#']
        for p in property:
            for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
                header.append(p + '.' + tr)
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
            for p in property:
                c1221, c1331, c2332 = 0, 0, 0
                for pair in aaPair:
                    if (pair[0] in group1[p] and pair[1] in group2[p]) or (
                            pair[0] in group2[p] and pair[1] in group1[p]):
                        c1221 = c1221 + 1
                        continue
                    if (pair[0] in group1[p] and pair[1] in group3[p]) or (
                            pair[0] in group3[p] and pair[1] in group1[p]):
                        c1331 = c1331 + 1
                        continue
                    if (pair[0] in group2[p] and pair[1] in group3[p]) or (
                            pair[0] in group3[p] and pair[1] in group2[p]):
                        c2332 = c2332 + 1
                code = code + [c1221 / len(aaPair), c1331 / len(aaPair), c2332 / len(aaPair)]
            encodings.append(code)
        return encodings

    def CTriad(self):
        AAGroup = {
            'g1': 'AGV',
            'g2': 'ILFP',
            'g3': 'YMTS',
            'g4': 'HNQW',
            'g5': 'RK',
            'g6': 'DE',
            'g7': 'C'
        }

        myGroups = sorted(AAGroup.keys())

        AADict = {}
        for g in myGroups:
            for aa in AAGroup[g]:
                AADict[aa] = g

        features = [f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

        encodings = []
        header = ['#']
        for f in features:
            header.append(f)
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            if len(sequence) < 3:
                print('Error: for "CTriad" encoding, the input fasta sequences should be greater than 3. \n\n')
                return 0
            code = code + CalculateKSCTriad(sequence, 0, features, AADict)
            encodings.append(code)

        return encodings

    def DDE(self):
        AA = 'ACDEFGHIKLMNPQRSTVWY'

        myCodons = {
            'A': 4,
            'C': 2,
            'D': 2,
            'E': 2,
            'F': 2,
            'G': 4,
            'H': 2,
            'I': 3,
            'K': 2,
            'L': 6,
            'M': 1,
            'N': 2,
            'P': 4,
            'Q': 2,
            'R': 6,
            'S': 6,
            'T': 4,
            'V': 4,
            'W': 1,
            'Y': 2
        }

        encodings = []
        diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
        header = ['#'] + diPeptides
        encodings.append(header)

        myTM = []
        for pair in diPeptides:
            myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

        AADict = {}
        for i in range(len(AA)):
            AADict[AA[i]] = i

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            tmpCode = [0] * 400
            for j in range(len(sequence) - 2 + 1):
                tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[
                    sequence[j + 1]]] + 1
            if sum(tmpCode) != 0:
                tmpCode = [i / sum(tmpCode) for i in tmpCode]

            myTV = []
            for j in range(len(myTM)):
                myTV.append(myTM[j] * (1 - myTM[j]) / (len(sequence) - 1))

            for j in range(len(tmpCode)):
                tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

            code = code + tmpCode
            encodings.append(code)
        return encodings

    def DPC(self):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        encodings = []
        diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
        header = ['#'] + diPeptides
        encodings.append(header)

        AADict = {}
        for i in range(len(AA)):
            AADict[AA[i]] = i

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            tmpCode = [0] * 400
            for j in range(len(sequence) - 2 + 1):
                tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[
                    sequence[j + 1]]] + 1
            if sum(tmpCode) != 0:
                tmpCode = [i / sum(tmpCode) for i in tmpCode]
            code = code + tmpCode
            encodings.append(code)
        return encodings

    def GAAC(self):
        group = {
            'alphatic': 'GAVLMI',
            'aromatic': 'FYW',
            'postivecharge': 'KRH',
            'negativecharge': 'DE',
            'uncharge': 'STCPNQ'
        }

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

    def GDPC(self):
        group = {
            'alphaticr': 'GAVLMI',
            'aromatic': 'FYW',
            'postivecharger': 'KRH',
            'negativecharger': 'DE',
            'uncharger': 'STCPNQ'
        }

        groupKey = group.keys()
        baseNum = len(groupKey)
        dipeptide = [g1 + '.' + g2 for g1 in groupKey for g2 in groupKey]

        index = {}
        for key in groupKey:
            for aa in group[key]:
                index[aa] = key

        encodings = []
        header = ['#'] + dipeptide
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])

            code = [name]
            myDict = {}
            for t in dipeptide:
                myDict[t] = 0

            sum = 0
            for j in range(len(sequence) - 2 + 1):
                myDict[index[sequence[j]] + '.' + index[sequence[j + 1]]] = myDict[index[sequence[j]] + '.' + index[
                    sequence[j + 1]]] + 1
                sum = sum + 1

            if sum == 0:
                for t in dipeptide:
                    code.append(0)
            else:
                for t in dipeptide:
                    code.append(myDict[t] / sum)
            encodings.append(code)

        return encodings

    def GTPC(self):
        group = {
            'alphaticr': 'GAVLMI',
            'aromatic': 'FYW',
            'postivecharger': 'KRH',
            'negativecharger': 'DE',
            'uncharger': 'STCPNQ'
        }

        groupKey = group.keys()
        baseNum = len(groupKey)
        triple = [g1 + '.' + g2 + '.' + g3 for g1 in groupKey for g2 in groupKey for g3 in groupKey]

        index = {}
        for key in groupKey:
            for aa in group[key]:
                index[aa] = key

        encodings = []
        header = ['#'] + triple
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])

            code = [name]
            myDict = {}
            for t in triple:
                myDict[t] = 0

            sum = 0
            for j in range(len(sequence) - 3 + 1):
                myDict[index[sequence[j]] + '.' + index[sequence[j + 1]] + '.' + index[sequence[j + 2]]] = myDict[index[
                                                                                                                      sequence[
                                                                                                                          j]] + '.' +
                                                                                                                  index[
                                                                                                                      sequence[
                                                                                                                          j + 1]] + '.' +
                                                                                                                  index[
                                                                                                                      sequence[
                                                                                                                          j + 2]]] + 1
                sum = sum + 1

            if sum == 0:
                for t in triple:
                    code.append(0)
            else:
                for t in triple:
                    code.append(myDict[t] / sum)
            encodings.append(code)

        return encodings

    def KSCTriad(self, gap=0):
        AAGroup = {
            'g1': 'AGV',
            'g2': 'ILFP',
            'g3': 'YMTS',
            'g4': 'HNQW',
            'g5': 'RK',
            'g6': 'DE',
            'g7': 'C'
        }

        myGroups = sorted(AAGroup.keys())

        AADict = {}
        for g in myGroups:
            for aa in AAGroup[g]:
                AADict[aa] = g

        features = [f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

        encodings = []
        header = ['#']
        for g in range(gap + 1):
            for f in features:
                header.append(f + '.gap' + str(g))
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            if len(sequence) < 2 * gap + 3:
                print(
                    'Error: for "KSCTriad" encoding, the input fasta sequences should be greater than (2*gap+3). \n\n')
                return 0
            code = code + CalculateKSCTriad(sequence, gap, features, AADict)
            encodings.append(code)

        return encodings

    def QSOrder(self, nlag=30, w=0.1):
        dataFile = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[
            0]) + r'\iFeaturedata\Schneider-Wrede.txt' if platform.system() == 'Windows' else re.sub('codes$', '',
                                                                                                     os.path.split(
                                                                                                         os.path.realpath(
                                                                                                             __file__))[
                                                                                                         0]) + '/iFeaturedata/Schneider-Wrede.txt'
        dataFile1 = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[
            0]) + r'\iFeaturedata\Grantham.txt' if platform.system() == 'Windows' else re.sub('codes$', '',
                                                                                              os.path.split(
                                                                                                  os.path.realpath(
                                                                                                      __file__))[
                                                                                                  0]) + '/iFeaturedata/Grantham.txt'

        AA = 'ACDEFGHIKLMNPQRSTVWY'
        AA1 = 'ARNDCQEGHILKMFPSTWYV'

        DictAA = {}
        for i in range(len(AA)):
            DictAA[AA[i]] = i

        DictAA1 = {}
        for i in range(len(AA1)):
            DictAA1[AA1[i]] = i

        with open(dataFile) as f:
            records = f.readlines()[1:]
        AADistance = []
        for i in records:
            array = i.rstrip().split()[1:] if i.rstrip() != '' else None
            AADistance.append(array)
        AADistance = np.array(
            [float(AADistance[i][j]) for i in range(len(AADistance)) for j in range(len(AADistance[i]))]).reshape(
            (20, 20))

        with open(dataFile1) as f:
            records = f.readlines()[1:]
        AADistance1 = []
        for i in records:
            array = i.rstrip().split()[1:] if i.rstrip() != '' else None
            AADistance1.append(array)
        AADistance1 = np.array(
            [float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape(
            (20, 20))

        encodings = []
        header = ['#']
        for aa in AA1:
            header.append('Schneider.Xr.' + aa)
        for aa in AA1:
            header.append('Grantham.Xr.' + aa)
        for n in range(1, nlag + 1):
            header.append('Schneider.Xd.' + str(n))
        for n in range(1, nlag + 1):
            header.append('Grantham.Xd.' + str(n))
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            arraySW = []
            arrayGM = []
            for n in range(1, nlag + 1):
                arraySW.append(
                    sum([AADistance[DictAA[sequence[j]]][DictAA[sequence[j + n]]] ** 2 for j in
                         range(len(sequence) - n)]))
                arrayGM.append(sum(
                    [AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]] ** 2 for j in
                     range(len(sequence) - n)]))
            myDict = {}
            for aa in AA1:
                myDict[aa] = sequence.count(aa)
            for aa in AA1:
                code.append(myDict[aa] / (1 + w * sum(arraySW)))
            for aa in AA1:
                code.append(myDict[aa] / (1 + w * sum(arrayGM)))
            for num in arraySW:
                code.append((w * num) / (1 + w * sum(arraySW)))
            for num in arrayGM:
                code.append((w * num) / (1 + w * sum(arrayGM)))
            encodings.append(code)
        return encodings

    def TPC(self):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        encodings = []
        triPeptides = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
        header = ['#'] + triPeptides
        encodings.append(header)

        AADict = {}
        for i in range(len(AA)):
            AADict[AA[i]] = i

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            tmpCode = [0] * 8000
            for j in range(len(sequence) - 3 + 1):
                tmpCode[AADict[sequence[j]] * 400 + AADict[sequence[j + 1]] * 20 + AADict[sequence[j + 2]]] = tmpCode[
                                                                                                                  AADict[
                                                                                                                      sequence[
                                                                                                                          j]] * 400 +
                                                                                                                  AADict[
                                                                                                                      sequence[
                                                                                                                          j + 1]] * 20 +
                                                                                                                  AADict[
                                                                                                                      sequence[
                                                                                                                          j + 2]]] + 1
            if sum(tmpCode) != 0:
                tmpCode = [i / sum(tmpCode) for i in tmpCode]
            code = code + tmpCode
            encodings.append(code)
        return encodings

    def SOCNumber(self, nlag=30):
        dataFile = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[
            0]) + r'\iFeaturedata\Schneider-Wrede.txt' if platform.system() == 'Windows' else re.sub('codes$', '',
                                                                                                     os.path.split(
                                                                                                         os.path.realpath(
                                                                                                             __file__))[
                                                                                                         0]) + '/iFeaturedata/Schneider-Wrede.txt'
        dataFile1 = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[
            0]) + r'\iFeaturedata\Grantham.txt' if platform.system() == 'Windows' else re.sub('codes$', '',
                                                                                              os.path.split(
                                                                                                  os.path.realpath(
                                                                                                      __file__))[
                                                                                                  0]) + '/iFeaturedata/Grantham.txt'
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        AA1 = 'ARNDCQEGHILKMFPSTWYV'

        DictAA = {}
        for i in range(len(AA)):
            DictAA[AA[i]] = i

        DictAA1 = {}
        for i in range(len(AA1)):
            DictAA1[AA1[i]] = i

        with open(dataFile) as f:
            records = f.readlines()[1:]
        AADistance = []
        for i in records:
            array = i.rstrip().split()[1:] if i.rstrip() != '' else None
            AADistance.append(array)
        AADistance = np.array(
            [float(AADistance[i][j]) for i in range(len(AADistance)) for j in range(len(AADistance[i]))]).reshape(
            (20, 20))

        with open(dataFile1) as f:
            records = f.readlines()[1:]
        AADistance1 = []
        for i in records:
            array = i.rstrip().split()[1:] if i.rstrip() != '' else None
            AADistance1.append(array)
        AADistance1 = np.array(
            [float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape(
            (20, 20))

        encodings = []
        header = ['#']
        for n in range(1, nlag + 1):
            header.append('Schneider.lag' + str(n))
        for n in range(1, nlag + 1):
            header.append('gGrantham.lag' + str(n))
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            for n in range(1, nlag + 1):
                code.append(sum(
                    [AADistance[DictAA[sequence[j]]][DictAA[sequence[j + n]]] ** 2 for j in
                     range(len(sequence) - n)]) / (
                                    len(sequence) - n))

            for n in range(1, nlag + 1):
                code.append(sum([AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]] ** 2 for j in
                                 range(len(sequence) - n)]) / (len(sequence) - n))
            encodings.append(code)
        return encodings

    def APAAC(self, lambdaValue=30, w=0.05):
        dataFile = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[
            0]) + r'\iFeaturedata\PAAC.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(
            os.path.realpath(__file__))[0]) + '/iFeaturedata/PAAC.txt'
        with open(dataFile) as f:
            records = f.readlines()
        AA = ''.join(records[0].rstrip().split()[1:])
        AADict = {}
        for i in range(len(AA)):
            AADict[AA[i]] = i
        AAProperty = []
        AAPropertyNames = []
        for i in range(1, len(records) - 1):
            array = records[i].rstrip().split() if records[i].rstrip() != '' else None
            AAProperty.append([float(j) for j in array[1:]])
            AAPropertyNames.append(array[0])

        AAProperty1 = []
        for i in AAProperty:
            meanI = sum(i) / 20
            fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
            AAProperty1.append([(j - meanI) / fenmu for j in i])

        encodings = []
        header = ['#']
        for i in AA:
            header.append('Pc1.' + i)
        for j in range(1, lambdaValue + 1):
            for i in AAPropertyNames:
                header.append('Pc2.' + i + '.' + str(j))
        encodings.append(header)
        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            theta = []
            for n in range(1, lambdaValue + 1):
                for j in range(len(AAProperty1)):
                    theta.append(
                        sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                             range(len(sequence) - n)]) / (len(sequence) - n))
            myDict = {}
            for aa in AA:
                myDict[aa] = sequence.count(aa)

            code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
            code = code + [w * value / (1 + w * sum(theta)) for value in theta]
            encodings.append(code)
        return encodings

    def Geary(self,
              props=['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102', 'CHOC760101', 'BIGC670101', 'CHAM810101',
                     'DAYM780201'],
              nlag=30):

        AA = 'ARNDCQEGHILKMFPSTWYV'
        fileAAidx = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[
            0]) + r'\iFeaturedata\AAidx.txt' if platform.system() == 'Windows' else sys.path[
                                                                                        0] + '/iFeaturedata/AAidx.txt'
        with open(fileAAidx) as f:
            records = f.readlines()[1:]
        myDict = {}
        for i in records:
            array = i.rstrip().split('\t')
            myDict[array[0]] = array[1:]

        AAidx = []
        AAidxName = []
        for i in props:
            if i in myDict:
                AAidx.append(myDict[i])
                AAidxName.append(i)
            else:
                print('"' + i + '" properties not exist.')
                return None

        AAidx1 = np.array([float(j) for i in AAidx for j in i])
        AAidx = AAidx1.reshape((len(AAidx), 20))

        propMean = np.mean(AAidx, axis=1)
        propStd = np.std(AAidx, axis=1)

        for i in range(len(AAidx)):
            for j in range(len(AAidx[i])):
                AAidx[i][j] = (AAidx[i][j] - propMean[i]) / propStd[i]

        index = {}
        for i in range(len(AA)):
            index[AA[i]] = i

        encodings = []
        header = ['#']
        for p in props:
            for n in range(1, nlag + 1):
                header.append(p + '.lag' + str(n))
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            N = len(sequence)
            for prop in range(len(props)):
                xmean = sum([AAidx[prop][index[aa]] for aa in sequence]) / N
                for n in range(1, nlag + 1):
                    if len(sequence) > nlag:
                        # if key is '-', then the value is 0
                        rn = (N - 1) / (2 * (N - n)) * ((sum([(AAidx[prop][index.get(sequence[j], 0)] - AAidx[prop][
                            index.get(sequence[j + n], 0)]) ** 2 for j in range(len(sequence) - n)])) / (
                                                            sum([(AAidx[prop][index.get(sequence[j], 0)] - xmean) ** 2
                                                                 for j in range(len(sequence))])))
                    else:
                        rn = 'NA'
                    code.append(rn)
            encodings.append(code)
        return encodings

    def Moran(self, props=['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102',
                           'CHOC760101', 'BIGC670101', 'CHAM810101', 'DAYM780201'],
              nlag=30):
        AA = 'ARNDCQEGHILKMFPSTWYV'
        fileAAidx = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[
            0]) + r'\iFeaturedata\AAidx.txt' if platform.system() == 'Windows' else sys.path[
                                                                                        0] + '/iFeaturedata/AAidx.txt'

        with open(fileAAidx) as f:
            records = f.readlines()[1:]
        myDict = {}
        for i in records:
            array = i.rstrip().split('\t')
            myDict[array[0]] = array[1:]

        AAidx = []
        AAidxName = []
        for i in props:
            if i in myDict:
                AAidx.append(myDict[i])
                AAidxName.append(i)
            else:
                print('"' + i + '" properties not exist.')
                return None

        AAidx1 = np.array([float(j) for i in AAidx for j in i])
        AAidx = AAidx1.reshape((len(AAidx), 20))

        propMean = np.mean(AAidx, axis=1)
        propStd = np.std(AAidx, axis=1)

        for i in range(len(AAidx)):
            for j in range(len(AAidx[i])):
                AAidx[i][j] = (AAidx[i][j] - propMean[i]) / propStd[i]

        index = {}
        for i in range(len(AA)):
            index[AA[i]] = i

        encodings = []
        header = ['#']
        for p in props:
            for n in range(1, nlag + 1):
                header.append(p + '.lag' + str(n))
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            N = len(sequence)
            for prop in range(len(props)):
                xmean = sum([AAidx[prop][index[aa]] for aa in sequence]) / N
                for n in range(1, nlag + 1):
                    if len(sequence) > nlag:
                        # if key is '-', then the value is 0
                        fenzi = sum([(AAidx[prop][index.get(sequence[j], 0)] - xmean) * (
                                AAidx[prop][index.get(sequence[j + n], 0)] - xmean) for j in
                                     range(len(sequence) - n)]) / (N - n)
                        fenmu = sum(
                            [(AAidx[prop][index.get(sequence[j], 0)] - xmean) ** 2 for j in range(len(sequence))]) / N
                        rn = fenzi / fenmu
                    else:
                        rn = 'NA'
                    code.append(rn)
            encodings.append(code)
        return encodings

    def NMBroto(self,
                props=['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102', 'CHOC760101', 'BIGC670101', 'CHAM810101',
                       'DAYM780201'], nlag=30):
        AA = 'ARNDCQEGHILKMFPSTWYV'
        fileAAidx = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[
            0]) + r'\iFeaturedata\AAidx.txt' if platform.system() == 'Windows' else sys.path[
                                                                                        0] + '/iFeaturedata/AAidx.txt'
        with open(fileAAidx) as f:
            records = f.readlines()[1:]
        myDict = {}
        for i in records:
            array = i.rstrip().split('\t')
            myDict[array[0]] = array[1:]

        AAidx = []
        AAidxName = []
        for i in props:
            if i in myDict:
                AAidx.append(myDict[i])
                AAidxName.append(i)
            else:
                print('"' + i + '" properties not exist.')
                return None

        AAidx1 = np.array([float(j) for i in AAidx for j in i])
        AAidx = AAidx1.reshape((len(AAidx), 20))
        pstd = np.std(AAidx, axis=1)
        pmean = np.average(AAidx, axis=1)

        for i in range(len(AAidx)):
            for j in range(len(AAidx[i])):
                AAidx[i][j] = (AAidx[i][j] - pmean[i]) / pstd[i]

        index = {}
        for i in range(len(AA)):
            index[AA[i]] = i

        encodings = []
        header = ['#']
        for p in props:
            for n in range(1, nlag + 1):
                header.append(p + '.lag' + str(n))
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            N = len(sequence)
            for prop in range(len(props)):
                for n in range(1, nlag + 1):
                    if len(sequence) > nlag:
                        # if key is '-', then the value is 0
                        rn = sum(
                            [AAidx[prop][index.get(sequence[j], 0)] * AAidx[prop][index.get(sequence[j + n], 0)] for j
                             in range(len(sequence) - n)]) / (N - n)
                    else:
                        rn = 'NA'
                    code.append(rn)
            encodings.append(code)
        return encodings

    def CKSAAP(self, gap=5):
        if gap < 0:
            print('Error: the gap should be equal or greater than zero' + '\n\n')
            return 0
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        encodings = []
        aaPairs = []
        for aa1 in AA:
            for aa2 in AA:
                aaPairs.append(aa1 + aa2)
        header = ['#']
        for g in range(gap + 1):
            for aa in aaPairs:
                header.append(aa + '.gap' + str(g))
        encodings.append(header)
        for i in self.fastaLi:
            name, sequence = i[0], i[1]
            code = [name]
            for g in range(gap + 1):
                myDict = {}
                for pair in aaPairs:
                    myDict[pair] = 0
                sum = 0
                for index1 in range(len(sequence)):
                    index2 = index1 + g + 1
                    if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                        index2] in AA:
                        myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                        sum = sum + 1
                for pair in aaPairs:
                    if sum == 0:
                        print(sequence)
                    else:
                        code.append(myDict[pair] / sum)
            encodings.append(code)
        return encodings

    def AAINDEX(self, **kw):
        AA = 'ARNDCQEGHILKMFPSTWYV'

        fileAAindex = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[
            0]) + r'\iFeaturedata\AAindex.txt' if platform.system() == 'Windows' else re.sub('codes$', '',
                                                                                             os.path.split(
                                                                                                 os.path.realpath(
                                                                                                     __file__))[
                                                                                                 0]) + '/iFeaturedata/AAindex.txt'
        with open(fileAAindex) as f:
            records = f.readlines()[1:]

        AAindex = []
        AAindexName = []
        for i in records:
            AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
            AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)

        index = {}
        for i in range(len(AA)):
            index[AA[i]] = i

        encodings = []
        header = ['#']
        for pos in range(1, len(self.fastaLi[0][1]) + 1):
            for idName in AAindexName:
                header.append('SeqPos.' + str(pos) + '.' + idName)
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], i[1]
            code = [name]
            for aa in sequence:
                if aa == '-':
                    for j in AAindex:
                        code.append(0)
                    continue
                for j in AAindex:
                    code.append(j[index[aa]])
            encodings.append(code)

        return encodings

    def BINARY(self, **kw):
        AA = 'ARNDCQEGHILKMFPSTWYV'
        encodings = []
        header = ['#']
        for i in range(1, len(self.fastaLi[0][1]) * 20 + 1):
            header.append('BINARY.F' + str(i))
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], i[1]
            code = [name]
            for aa in sequence:
                if aa == '-':
                    code = code + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    continue
                for aa1 in AA:
                    tag = 1 if aa == aa1 else 0
                    code.append(tag)
            encodings.append(code)
        return encodings

    def PAAC(self, lambdaValue=30, w=0.05):

        dataFile = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[
            0]) + r'\iFeaturedata\PAAC.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(
            os.path.realpath(__file__))[0]) + '/iFeaturedata/PAAC.txt'
        with open(dataFile) as f:
            records = f.readlines()
        AA = ''.join(records[0].rstrip().split()[1:])
        AADict = {}
        for i in range(len(AA)):
            AADict[AA[i]] = i
        AAProperty = []
        AAPropertyNames = []
        for i in range(1, len(records)):
            array = records[i].rstrip().split() if records[i].rstrip() != '' else None
            AAProperty.append([float(j) for j in array[1:]])
            AAPropertyNames.append(array[0])

        AAProperty1 = []
        for i in AAProperty:
            meanI = sum(i) / 20
            fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
            AAProperty1.append([(j - meanI) / fenmu for j in i])

        encodings = []
        header = ['#']
        for aa in AA:
            header.append('Xc1.' + aa)
        for n in range(1, lambdaValue + 1):
            header.append('Xc2.lambda' + str(n))
        encodings.append(header)

        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = [name]
            theta = []
            for n in range(1, lambdaValue + 1):
                theta.append(
                    sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in
                         range(len(sequence) - n)]) / (
                            len(sequence) - n))
            myDict = {}
            for aa in AA:
                myDict[aa] = sequence.count(aa)
            code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
            code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
            encodings.append(code)
        return encodings
