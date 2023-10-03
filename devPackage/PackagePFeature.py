import pandas as pd
import numpy as np
import csv
import sys, re, os
from collections import Counter
import math
std = 'ARNDCQEGHILKMFPSTWYV'


def entropy_single(seq):
    seq = seq.upper()
    num, length = Counter(seq), len(seq)
    return -sum(freq / length * math.log(freq / length, 2) for freq in num.values())


class PFeature:
    def __init__(self, seqDict, featureDict):
        self.fastaLi = seqDict.items()
        self.seqsNameLi = list(seqDict.keys())
        self.seqs = list(seqDict.values())
        self.featureDf = pd.DataFrame()
        if featureDict["DDOR"] is True:
            DDORDf = pd.DataFrame(self.DDOR())
            self.featureDf = pd.concat([self.featureDf, DDORDf], axis=1)

        if featureDict["RRI"] is True:
            RRIDf = pd.DataFrame(self.RRI())
            self.featureDf = pd.concat([self.featureDf, RRIDf], axis=1)

        if featureDict["SER"] is True:
            SERDf = pd.DataFrame(self.SER())
            self.featureDf = pd.concat([self.featureDf, SERDf], axis=1)

        if featureDict["SEP"] is True:
            SEPDf = pd.DataFrame(self.SEP())
            self.featureDf = pd.concat([self.featureDf, SEPDf], axis=1)

        if featureDict["SE"] is True:
            SEDf = pd.DataFrame(self.SE())
            self.featureDf = pd.concat([self.featureDf, SEDf], axis=1)

        if featureDict["QSO"][0] is True:
            QSODf = pd.DataFrame(self.QSO(gap=featureDict["QSO"][1], w=featureDict["QSO"][2]))
            self.featureDf = pd.concat([self.featureDf, QSODf], axis=1)

        else:
            pass

    def getOutputDf(self):
        self.featureDf.index = self.seqsNameLi
        return self.featureDf

    std = 'ARNDCQEGHILKMFPSTWYV'

    def DDOR(self, out="../devPackage/intermediate files/DDOR.csv"):
        nameList = []
        sequenceList = []
        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            nameList.append(name)
            sequenceList.append(sequence)
        f = open(out, 'w')
        sys.stdout = f
        for i in std:
            print('DDR_' + i, end=",")
        print("")
        for i in range(0, len(sequenceList)):
            s = sequenceList[i]
            p = s[::-1]
            for j in std:
                zz = ([pos for pos, char in enumerate(s) if char == j])
                pp = ([pos for pos, char in enumerate(p) if char == j])
                ss = []
                for i in range(0, (len(zz) - 1)):
                    ss.append(zz[i + 1] - zz[i] - 1)
                if zz == []:
                    ss = []
                else:
                    ss.insert(0, zz[0])
                    ss.insert(len(ss), pp[0])
                cc1 = (sum([e for e in ss]) + 1)
                cc = sum([e * e for e in ss])
                zz2 = cc / cc1
                print("%.2f" % zz2, end=",")
            print("")
        f.truncate()
        f.close()
        sys.stdout = sys.__stdout__
        df = pd.read_csv("../devPackage/intermediate files/DDOR.csv", index_col=[-1]).reset_index(drop=True)
        return df

    def RRI(self, out="../devPackage/intermediate files/RRI.csv"):
        nameList = []
        sequenceList = []
        for i in self.fastaLi:
            name, sequence = i[0], re.sub('-', '', i[1])
            nameList.append(name)
            sequenceList.append(sequence)
        count = 0
        cc = []
        i = 0
        x = 0
        temp = pd.DataFrame()
        f = open(out, 'w')
        sys.stdout = f
        print(
            "RRI_A,RRI_C,RRI_D,RRI_E,RRI_F,RRI_G,RRI_H,RRI_I,RRI_K,RRI_L,RRI_M,RRI_N,RRI_P,RRI_Q,RRI_R,RRI_S,RRI_T,RRI_V,RRI_W,RRI_Y,")
        for q in range(0, len(sequenceList)):
            while i < len(std):
                cc = []
                for j in sequenceList[q]:
                    if j == std[i]:
                        count += 1
                        cc.append(count)
                    else:
                        count = 0
                while x < len(cc):
                    if x + 1 < len(cc):
                        if cc[x] != cc[x + 1]:
                            if cc[x] < cc[x + 1]:
                                cc[x] = 0
                    x += 1
                cc1 = [e for e in cc if e != 0]
                cc = [e * e for e in cc if e != 0]
                zz = sum(cc)
                zz1 = sum(cc1)
                if zz1 != 0:
                    zz2 = zz / zz1
                else:
                    zz2 = 0
                print("%.2f" % zz2, end=',')
                i += 1
            i = 0
            print(" ")
        f.truncate()
        sys.stdout = sys.__stdout__
        df = pd.read_csv("../devPackage/intermediate files/RRI.csv", index_col=[-1]).reset_index(drop=True)
        return df

    def SER(self, out="../devPackage/intermediate files/SER.csv"):
        data = list(self.seqs)
        Val = np.zeros(len(data))
        GH = []
        for i in range(len(data)):
            my_list = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0,
                       'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}
            data1 = ''
            data1 = str(data[i])
            data1 = data1.upper()
            allowed = set(('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'))
            is_data_invalid = set(data1).issubset(allowed)
            if is_data_invalid == False:
                print("Error: Please check for invalid inputs in the sequence.", "\nError in: ", "Sequence cutoff=",
                      i + 1, ",", "Sequence = ", data[i], ",",
                      "\nNOTE: Spaces, Special characters('[@_!#$%^&*()<>?/\|}{~:]') and Extra characters(BJOUXZ) should not be there.")
                return
            seq = data[i]
            seq = seq.upper()
            num, length = Counter(seq), len(seq)
            num = dict(sorted(num.items()))
            C = list(num.keys())
            F = list(num.values())
            for key, value in my_list.items():
                for j in range(len(C)):
                    if key == C[j]:
                        my_list[key] = round(((F[j] / length) * math.log(F[j] / length, 2)), 3)
            GH.append(list(my_list.values()))
        file = open(out, 'w', newline='')  # output file
        with file:
            writer = csv.writer(file);
            writer.writerow(('SER_A', 'SER_C', 'SER_D', 'SER_E', 'SER_F', 'SER_G', 'SER_H', 'SER_I', 'SER_K', 'SER_L',
                             'SER_M', 'SER_N', 'SER_P', 'SER_Q', 'SER_R', 'SER_S', 'SER_T', 'SER_V', 'SER_W', 'SER_Y'));
            writer.writerows(GH);
        df = pd.read_csv("../devPackage/intermediate files/SER.csv").reset_index(drop=True)
        return df

    def SEP(self, out="../devPackage/intermediate files/SEP.csv"):
        data = list(self.seqs)
        GH = []
        for i in range(len(data)):
            my_list = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0,
                       'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}
            data1 = ''
            data1 = str(data[i])
            data1 = data1.upper()
            allowed = set(('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'))
            is_data_invalid = set(data1).issubset(allowed)
            if is_data_invalid == False:
                print("Error: Please check for invalid inputs in the sequence.", "\nError in: ", "Sequence cutoff=",
                      i + 1, ",", "Sequence = ", data[i], ",",
                      "\nNOTE: Spaces, Special characters('[@_!#$%^&*()<>?/\|}{~:]') and Extra characters(BJOUXZ) should not be there.")
                return
            seq = data[i]
            seq = seq.upper()
            num, length = Counter(seq), len(seq)
            num = dict(sorted(num.items()))
            C = list(num.keys())
            F = list(num.values())
            for key, value in my_list.items():
                for j in range(len(C)):
                    if key == C[j]:
                        my_list[key] = round(((F[j] / length) * math.log(F[j] / length, 2)), 3)
            GH.append(list(my_list.values()))
        file = open(out, 'w', newline='')  # output file
        with file:
            writer = csv.writer(file);
            writer.writerow(
                ('SEP_A', 'SEP_C', 'SEP_D', 'SEP_E', 'SEP_F', 'SEP_G', 'SEP_H', 'SEP_I', 'SEP_K', 'SEP_L', 'SEP_M', 'SEP_N', 'SEP_P', 'SEP_Q', 'SEP_R', 'SEP_S', 'SEP_T', 'SEP_V', 'SEP_W', 'SEP_Y'));
            writer.writerows(GH);
        df = pd.read_csv("../devPackage/intermediate files/SEP.csv").reset_index(drop=True)
        return df

    def SE(self, out="../devPackage/intermediate files/SE.csv"):
        data = list(self.seqs)
        Val = []
        header = ["Shannon-Entropy"]
        for i in range(len(data)):
            data1 = ''
            data1 = str(data[i])
            data1 = data1.upper()
            allowed = set(('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'))
            is_data_invalid = set(data1).issubset(allowed)
            if is_data_invalid == False:
                print("Error: Please check for invalid inputs in the sequence.", "\nError in: ", "Sequence cutoff=",
                      i + 1, ",", "Sequence = ", data[i], ",",
                      "\nNOTE: Spaces, Special characters('[@_!#$%^&*()<>?/\|}{~:]') and Extra characters(BJOUXZ) should not be there.")
                return
            Val.append(round((entropy_single(str(data[i]))), 3))
            # print(Val[i])
        file = open(out, 'w', newline='\n')  # output file
        with file:
            writer = csv.writer(file, delimiter='\n');
            writer.writerow(header)
            writer.writerow(Val);
        df = pd.read_csv("../devPackage/intermediate files/SE.csv").reset_index(drop=True)
        return df

    def QSO(self, gap, out="../devPackage/intermediate files/QSO.csv", w=0.1):
        ff = []
        seqs = self.seqs
        for i in range(0, len(seqs)):
            ff.append(len(seqs[i]))
        if min(ff) < gap:
            print("Error: All sequences' length should be higher than :", gap)
        else:
            mat1 = pd.read_csv("../devPackage/PFeaturedata/Schneider-Wrede.csv", index_col='Name')
            mat2 = pd.read_csv("../devPackage/PFeaturedata/Grantham.csv", index_col='Name')
            s1 = []
            s2 = []
            for i in range(0, len(seqs)):
                for n in range(1, gap + 1):
                    sum1 = 0
                    sum2 = 0
                    for j in range(0, (len(seqs[i]) - n)):
                        sum1 = sum1 + (mat1[seqs[i][j]][seqs[i][j + n]]) ** 2
                        sum2 = sum2 + (mat2[seqs[i][j]][seqs[i][j + n]]) ** 2
                    s1.append(sum1)
                    s2.append(sum2)
            zz = pd.DataFrame(np.array(s1).reshape(len(seqs), gap))
            zz["sum"] = zz.sum(axis=1)
            zz2 = pd.DataFrame(np.array(s2).reshape(len(seqs), gap))
            zz2["sum"] = zz2.sum(axis=1)
            c1 = []
            c2 = []
            c3 = []
            c4 = []
            h1 = []
            h2 = []
            h3 = []
            h4 = []
            for aa in std:
                h1.append('QSO' + str(gap) + '_SC_' + aa)
            for aa in std:
                h2.append('QSO' + str(gap) + '_G_' + aa)
            for n in range(1, gap + 1):
                h3.append('SC' + str(n))
            h3 = ['QSO' + str(gap) + '_' + sam for sam in h3]
            for n in range(1, gap + 1):
                h4.append('G' + str(n))
            h4 = ['QSO' + str(gap) + '_' + sam for sam in h4]
            for i in range(0, len(seqs)):
                AA = {}
                for j in std:
                    AA[j] = seqs[i].count(j)
                    c1.append(AA[j] / (1 + w * zz['sum'][i]))
                    c2.append(AA[j] / (1 + w * zz2['sum'][i]))
                for k in range(0, gap):
                    c3.append((w * zz[k][i]) / (1 + w * zz['sum'][i]))
                    c4.append((w * zz[k][i]) / (1 + w * zz['sum'][i]))
            pp1 = np.array(c1).reshape(len(seqs), len(std))
            pp2 = np.array(c2).reshape(len(seqs), len(std))
            pp3 = np.array(c3).reshape(len(seqs), gap)
            pp4 = np.array(c4).reshape(len(seqs), gap)
            zz5 = round(pd.concat(
                [pd.DataFrame(pp1, columns=h1), pd.DataFrame(pp2, columns=h2), pd.DataFrame(pp3, columns=h3),
                 pd.DataFrame(pp4, columns=h4)], axis=1), 4)
            zz5.to_csv(out, index=None, encoding='utf-8')
        df = pd.read_csv("../devPackage/intermediate files/QSO.csv").reset_index(drop=True)
        return df




