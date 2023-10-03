import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, matthews_corrcoef, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt


class Scoring:
    def __init__(self, predVectorList, probVectorList, answerDf, modelNameList=None):
        """

        :param predVectorList: 一個 preditor 會產生一組 (0 or 1) ndarray, 多個 ndarray 會形成一個 list, 會同時算分
        :param probVectorList: 一個 preditor 會產生一組 probability ndarray, 多個 ndarray 會形成一個 list, 會同時算分
        :param answerDf: test dataDf 的答案 1 維的 dataframe
        :param modelNameList: model 名稱, 會顯示在表格裡面
        """
        self.predVectorList = predVectorList
        self.probVectorList = probVectorList
        self.predVectorList_cutOff = None
        self.answerDf = answerDf
        self.modelNameList = []
        self.bestCutOffList = None
        if modelNameList is None:
            modelNameList = ['rbfsvm', 'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et', 'lightgbm', 'rf', 'xgboost', 'gpc', 'mlp', 'dt', 'svm', 'qda']
        for modelName in modelNameList:
            self.modelNameList.append(modelName.lower())
        if len(self.predVectorList) != len(self.modelNameList) or len(self.probVectorList) != len(self.modelNameList):
            print('predVectorList length not equal to modelNameList length')

    def doScoring(self, b_optimizedMcc=False, path=None, sortColumn='mcc'):
        """
        AUC用probVector計算,其餘分數用predVector計算
        path: the path of scoreDf to csv.
        sortColumn: sort value by column name, ex: accuracy, precision, recall, f1_score, auc, specificity, mcc, bestCutoff.
        :return:
        """
        scoreList = []
        if b_optimizedMcc:
            predVectorList = self.predVectorList_cutOff
            bestCutOffList = self.bestCutOffList
        else:
            predVectorList = self.predVectorList
            bestCutOffList = [0.5] * len(self.modelNameList)
        for (predVector, probVector, bestCutOff) in zip(predVectorList, self.probVectorList, bestCutOffList):
            fpr, tpr, threshold = roc_curve(self.answerDf, probVector)
            auc1 = auc(fpr, tpr)
            if len(confusion_matrix(self.answerDf, predVector).ravel()) < 4:
                specificity = None
            else:
                tn, fp, fn, tp = confusion_matrix(self.answerDf, predVector).ravel()
                specificity = tn / (tn + fp)
            scoreDict = {"accuracy": accuracy_score(self.answerDf, predVector),
                         "precision": precision_score(self.answerDf, predVector),
                         "recall": recall_score(self.answerDf, predVector),
                         "f1_score": f1_score(self.answerDf, predVector),
                         "auc": auc1,
                         "specificity": specificity,
                         "mcc": matthews_corrcoef(self.answerDf, predVector),
                         "bestCutoff": bestCutOff}
            scoreList.append(scoreDict)
        scoreDf = pd.DataFrame(scoreList, index=[self.modelNameList])
        if sortColumn is not None:
            scoreDf = scoreDf.sort_values(by=sortColumn, ascending=False)
        if path is not None:
            scoreDf.to_csv(path)
        return scoreDf

    def optimizeMcc(self, cutOffList=None, method='mcc'):
        if cutOffList is None:
            cutOffList = [0.5]
        probDf = pd.DataFrame(self.probVectorList).T
        probDf.columns = self.modelNameList
        scoreDf = pd.DataFrame(index=cutOffList)
        predArrDf = pd.DataFrame(index=cutOffList)
        for modelName in self.modelNameList:
            scoreList = []
            predArrList = []
            for cutOff in cutOffList:
                probDf.loc[probDf[modelName] > cutOff, 'predVector_' + str(modelName)] = 1
                probDf.loc[probDf[modelName] <= cutOff, 'predVector_' + str(modelName)] = 0
                predArr = np.array(probDf['predVector_' + str(modelName)].values.tolist())
                predArrList.append(predArr)
                if method == 'mcc':
                    mcc = matthews_corrcoef(self.answerDf, predArr)
                    scoreList.append(mcc)
                elif method == 'acc':
                    acc = accuracy_score(self.answerDf, predArr)
                    scoreList.append(acc)
                elif method == 'auc':
                    fpr, tpr, threshold = roc_curve(self.answerDf, predArr)
                    auc1 = auc(fpr, tpr)
                    scoreList.append(auc1)
            scoreDf[modelName] = scoreList
            predArrDf[modelName] = predArrList
        mccMaxSeries = scoreDf.idxmax(axis=0)
        mccMaxList = mccMaxSeries.tolist()
        bestPredArrList = []
        for (modelName, mccMax) in zip(self.modelNameList, mccMaxList):
            bestPredArr = predArrDf[modelName].loc[mccMax]
            bestPredArrList.append(bestPredArr)
        self.bestCutOffList = mccMaxList
        self.predVectorList_cutOff = bestPredArrList

    @staticmethod
    def plotPredConfidence(predictionsList, trueLabelsDf, numBins=10, modelNameList=None, outputExcel=None, figSave=False, figSavePath=None):
        trueLabelsList = trueLabelsDf['y'].to_list()

        with pd.ExcelWriter(outputExcel, engine='openpyxl', mode='w') as writer:
            data = {
                'Prediction Probability': [f'{i / numBins:.1f}-{(i + 1) / numBins:.1f}' for i in range(numBins)]}

            numPositiveData = {
                'Prediction Probability': [f'{i / numBins:.1f}-{(i + 1) / numBins:.1f}' for i in range(numBins)]}

            totalSamplesData = {
                'Prediction Probability': [f'{i / numBins:.1f}-{(i + 1) / numBins:.1f}' for i in range(numBins)]}

            for i, predictions in enumerate(predictionsList):
                # 將預測值轉換為NumPy數組
                predictionsArr = np.array(predictions)

                # 將真實標籤轉換為NumPy數組
                trueLabelsArr = np.array(trueLabelsList)

                # 初始化y軸數據為0
                yValues = np.zeros(numBins)
                numPositiveList = []
                totalSamplesList = []

                # 遍歷每個區間
                for j in range(numBins):
                    # 獲取當前區間的上下界
                    lowerBound = j / numBins
                    upperBound = (j + 1) / numBins

                    # 將預測值為1.0的記錄在0.9-1.0區間
                    if upperBound == 1.0:
                        mask = (predictionsArr >= lowerBound) & (predictionsArr <= upperBound)
                    else:
                        mask = (predictionsArr >= lowerBound) & (predictionsArr < upperBound)

                    # 在該區間內計算真實positive的比例
                    numPositive = np.sum(trueLabelsArr[mask] == 1)
                    totalSamples = np.sum(mask)
                    numPositiveList.append(numPositive)
                    totalSamplesList.append(totalSamples)

                    # 計算並記錄y軸數據
                    if totalSamples > 0:
                        yValues[j] = numPositive / totalSamples


                data[modelNameList[i]] = yValues
                numPositiveData[modelNameList[i]] = numPositiveList
                totalSamplesData[modelNameList[i]] = totalSamplesList

                # 繪製圖形
                xValues = np.arange(numBins)

                plt.figure(figsize=(12, 9), dpi=300)
                plt.plot(xValues, yValues, marker='o')
                plt.xlabel('Prediction Probability')
                plt.ylabel('True Positive Rate')
                plt.title(f'Confidence Analysis - {modelNameList[i]}')

                plt.xticks(xValues, [f'{i / numBins:.1f}-{(i + 1) / numBins:.1f}' for i in range(numBins)])  # 設定x軸刻度

                if figSave:
                    plt.savefig(figSavePath + f'plotPredConfidence_{modelNameList[i]}.png')

                plt.show()

            if outputExcel:
                df1 = pd.DataFrame(data)
                df2 = pd.DataFrame(numPositiveData)
                df3 = pd.DataFrame(totalSamplesData)
                sheetData = {'plotPredConfidence': df1,
                             'numPositive': df2,
                             'totalSamples': df3}

                for sheet_name, df in sheetData.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
