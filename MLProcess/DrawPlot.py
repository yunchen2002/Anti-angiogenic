import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve, matthews_corrcoef
import numpy as np
from scipy.stats import pearsonr


class DrawPlot:
    def __init__(self, answerDf, modelList, modelNameList=None, predArrList=None, probArrList=None):
        self.predArrList = predArrList  # 0 or 1 的預測結果
        self.probArrList = probArrList  # 機率的預測結果
        self.answerDf = answerDf
        self.modelList = modelList
        if modelNameList is None:
            modelNameList = ['rbfsvm', 'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et', 'lightgbm', 'rf', 'xgboost', 'gpc', 'mlp', 'dt', 'svm', 'qda']
        self.modelNameList = modelNameList

    def drawROC(self, colorList=None, title=False,
                titleName='Receiver Operating Characteristic', setDpi=True,
                legendSize=11, labelSize=20, save=False,
                saveLoc='Multi_Single_Model.png', show=True,
                dpi=300, figSize=(12, 9), topNum=None):
        """

        :param topNum:
        :param colorList: input color list
        :param title:
        :param titleName:
        :param setDpi:
        :param legendSize:
        :param labelSize:
        :param save:
        :param saveLoc:
        :param show:
        :param dpi:
        :param figSize:
        :return:
        """
        if colorList is None:
            colorList = ['black', 'gray', 'deeppink', 'lightcoral',
                         'brown', 'red', 'peru', 'orange',
                         'gold', 'yellow', 'olivedrab', 'yellowgreen',
                         'green', 'cyan', 'steelblue', 'blue',
                         'violet', 'purple']
            colorList = random.sample(colorList, k=len(self.modelList))
        aucList = []
        fprList = []
        tprList = []
        for probVector in self.probArrList:
            fpr, tpr, threshold = roc_curve(self.answerDf, probVector)
            auc1 = auc(fpr, tpr)
            aucList.append(auc1)
            fprList.append(fpr)
            tprList.append(tpr)
        aucDf = pd.DataFrame({'AUC': aucList, 'fpr': fprList, 'tpr': tprList}, index=self.modelNameList)
        aucDf = aucDf.sort_values(['AUC'], ascending=False)
        if topNum is None:
            topNum = len(aucDf.index.tolist())
        aucDfHead = aucDf.head(topNum)
        if setDpi:
            plt.figure(figsize=figSize, dpi=dpi)
        for (fprValue, tprValue, color, name, aucValue) in zip(aucDfHead['fpr'].values.tolist(), aucDfHead['tpr'].values.tolist(), colorList, aucDfHead.index.tolist(), aucDfHead['AUC'].tolist()):
            if title:
                plt.title(titleName)
            plt.plot(fprValue, tprValue, color=color, label='{} (AUC={:.3f}'.format(name, aucValue))
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate', fontsize=labelSize)
        plt.xlabel('False Positive Rate', fontsize=labelSize)
        plt.xticks(fontsize=labelSize)
        plt.yticks(fontsize=labelSize)
        plt.legend(fontsize=legendSize)
        plt.legend(loc='lower right')
        if save:
            plt.savefig(saveLoc)
        if show:
            plt.show()

        return aucDf

    def drawHeatmap(self, color="RdBu_r", setDpi=True, figSize=(12, 9), dpi=300, save=False, show=True,
                    saveLoc="Heatmap.png", isPcc=True):
        if isPcc:
            target = len(self.probArrList) * len(self.probArrList)
            vectorX = 0
            vectorY = 0
            times = 0
            simList = []
            while target > times:
                similarityArray = pearsonr(self.probArrList[vectorX], self.probArrList[vectorY])
                similarity = similarityArray[0]
                vectorY += 1
                times += 1
                simList.append(similarity)
                if vectorY == len(self.predArrList):
                    vectorX += 1
                    vectorY = 0
            simArray = np.array(simList).reshape(len(self.probArrList), len(self.probArrList))
            simDf = pd.DataFrame(simArray, index=self.modelNameList, columns=self.modelNameList)
            if setDpi:
                plt.figure(figsize=figSize, dpi=dpi)
            sns.set(font_scale=1.3)
            sns.heatmap(simDf, cmap=color)
            if save:
                plt.savefig(saveLoc)
            if show:
                plt.show()
        else:
            target = len(self.predArrList) * len(self.predArrList)
            vectorX = 0
            vectorY = 0
            times = 0
            simList = []
            while target > times:
                similarity = matthews_corrcoef(self.predArrList[vectorX], self.predArrList[vectorY])
                vectorY += 1
                times += 1
                simList.append(similarity)
                if vectorY == len(self.predArrList):
                    vectorX += 1
                    vectorY = 0
            simArray = np.array(simList).reshape(len(self.predArrList), len(self.predArrList))
            simDf = pd.DataFrame(simArray, index=self.modelNameList, columns=self.modelNameList)
            if setDpi:
                plt.figure(figsize=figSize, dpi=dpi)
            sns.set(font_scale=1.3)
            sns.heatmap(simDf, cmap=color)
            if save:
                plt.savefig(saveLoc)
            if show:
                plt.show()

        return simDf
