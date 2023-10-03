import pandas as pd


class centerGDP:
    def __init__(self, seqDict, featureDict):
        self.b_start = featureDict["Usage"]
        if self.b_start is True:
            if featureDict["UseGap"] is True:
                AA = 'ARNDCQEGHILKMFPSTWYV-'
                self.diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA if aa1 != '-']
            else:
                AA = 'ARNDCQEGHILKMFPSTWYV'
                self.diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
            fastaLi = seqDict.values()
            self.seqsNameLi = list(seqDict.keys())
            self.featureDf = pd.DataFrame()
            for i in fastaLi:
                centerGDPDf = self.count_aa(i, featureDict["gap_size"])
                self.featureDf = pd.concat([self.featureDf, centerGDPDf], axis=0)
        else:
            pass

    def getOutputDf(self):
        if self.b_start is True:
            self.featureDf.index = self.seqsNameLi
            return self.featureDf
        else:
            pass

    def count_aa(self, sequence, gap_size=-1):
        aa_dict = {}
        diPeptides = self.diPeptides
        for s in diPeptides:
            aa_dict.setdefault(s, 0)
        center = len(sequence) // 2

        if gap_size == -1:
            # process the whole sequence
            for i in range(len(sequence)):
                if i != center:
                    aa_dict[sequence[center] + sequence[i]] += 1
        else:
            # process a sub-sequence with given gap_size
            start = max(center - gap_size, 0)
            end = min(center + gap_size + 1, len(sequence))
            for i in range(start, end):
                if i != center:
                    aa_dict[sequence[center] + sequence[i]] += 1
        aa_df = pd.DataFrame([aa_dict])
        return aa_df

