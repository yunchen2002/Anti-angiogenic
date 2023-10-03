import pandas as pd
import numpy as np


class Voting:
    def __init__(self, dataX, modelList):
        self.modelList = modelList
        self.dataX = dataX
        if len(self.modelList) % 2 == 0:
            print('Using even cutoff of predictors. May cause the situation of equal positive and negative votes, in which the prediction will be negative.')

    def vote_predVector(self):
        """
        使用 predict 出的 (0 or 1) 做 voting。
        預設只能用奇數個方法進行投票, 若用偶數個方法, 則同票數會預測 negative。
        """
        predArrList = []
        for model in self.modelList:
            predArr = model.predict(self.dataX)
            predArrList.append(predArr)
        predArrDf = pd.DataFrame(predArrList)
        predArrMeanDf = pd.DataFrame(predArrDf.mean(0), columns=['predVector'])
        predArrMeanDf.loc[predArrMeanDf['predVector'] > 0.5, 'vote_predVector'] = int(1)
        predArrMeanDf.loc[predArrMeanDf['predVector'] <= 0.5, 'vote_predVector'] = int(0)
        votingPredArr = np.array(predArrMeanDf['vote_predVector'].values.tolist())

        return votingPredArr

    def vote_probVector(self, cutoff=0.5):
        '''
        用每個model預測的prob來投票
        :param cutoff:
        :return: [votingPredArr]投票完的結果根據cutoff轉成[0 & 1], [votingProbArr]投票完的結果以小數來呈現
        '''
        probArrList = []
        for model in self.modelList:
            if model.__class__.__name__ == 'RidgeClassifier':
                vector = model.predict(self.dataX)
                probArrList.append(vector)
            elif model.__class__.__name__ == 'SGDClassifier':
                vector = model.predict(self.dataX)
                probArrList.append(vector)
            else:
                vector = model.predict_proba(self.dataX)
                probArrList.append(vector[:, 1])
        probArrDf = pd.DataFrame(probArrList)
        probArrMeanDf = pd.DataFrame(probArrDf.mean(0), columns=['probVector'])
        probArrMeanDf.loc[probArrMeanDf['probVector'] > cutoff, 'vote_predVector'] = 1
        probArrMeanDf.loc[probArrMeanDf['probVector'] <= cutoff, 'vote_predVector'] = 0
        votingProbArr = np.array(probArrMeanDf['probVector'].values.tolist())
        votingPredArr = np.array(probArrMeanDf['vote_predVector'].values.tolist())

        return votingPredArr, votingProbArr
