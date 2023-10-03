import random
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import os



class Normalization:
    def __init__(self, trainDf, testDf):
        if trainDf is not None:
            self.trainDfIndex = trainDf.index.to_list()
            self.trainDfAnswer = trainDf[['y']]
            trainDfFeature = trainDf.drop('y', axis=1)
            self.trainDfFeatureCol = trainDfFeature.columns.to_list()
            self.trainArray = trainDfFeature.values
        # ==============================================================
        if testDf is not None:
            self.testDfIndex = testDf.index.to_list()
            self.testDfAnswer = testDf[['y']]
            testDfFeature = testDf.drop('y', axis=1)
            self.testDfFeatureCol = testDfFeature.columns.to_list()
            self.testArray = testDfFeature.values
        self.nmlzParams = None
        self.standardSca = None
        self.minMaxSca = None
        self.robustSca = None

    #  改存 scaler 並測試
    def standard(self):
        self.standardSca = StandardScaler()
        scalerDf = self.standardSca.fit_transform(self.trainArray)
        # self.nmlzParams = scaler.get_params()
        scalerDf = pd.DataFrame(scalerDf, index=self.trainDfIndex, columns=self.trainDfFeatureCol)
        scalerDf.insert(scalerDf.shape[1], "y", self.trainDfAnswer)

        return scalerDf, self.standardSca

    def standardTest(self, loadParams=False, loadNmlzParamsPklPath=None):
        # rs = StandardScaler()
        if loadParams:
            with open(loadNmlzParamsPklPath, 'rb') as f:
                standardScaler = pickle.load(f)
                scalerDf = standardScaler.fit_transform(self.testArray)
        else:
            scalerDf = self.standardSca.fit_transform(self.testArray)
        scalerDf = pd.DataFrame(scalerDf, index=self.testDfIndex, columns=self.testDfFeatureCol)
        scalerDf.insert(scalerDf.shape[1], "y", self.testDfAnswer)
        return scalerDf

    def minMax(self):
        self.minMaxSca = MinMaxScaler()
        scalerDf = self.minMaxSca.fit_transform(self.trainArray)
        # self.nmlzParams = scaler.get_params()
        scalerDf = pd.DataFrame(scalerDf, index=self.trainDfIndex, columns=self.trainDfFeatureCol)
        scalerDf.insert(scalerDf.shape[1], "y", self.trainDfAnswer)

        return scalerDf, self.minMaxSca

    def minMaxTest(self, loadParams=False, loadNmlzParamsPklPath=None):
        # rs = MinMaxScaler()
        if loadParams:
            with open(loadNmlzParamsPklPath, 'rb') as f:
                minMaxScaler = pickle.load(f)
                scalerDf = minMaxScaler.fit_transform(self.testArray)
        else:
            scalerDf = self.minMaxSca.fit_transform(self.testArray)
        scalerDf = pd.DataFrame(scalerDf, index=self.testDfIndex, columns=self.testDfFeatureCol)
        scalerDf.insert(scalerDf.shape[1], "y", self.testDfAnswer)
        return scalerDf

    def robust(self):
        self.robustSca = RobustScaler()
        scalerDf = self.robustSca.fit_transform(self.trainArray)
        # self.nmlzParams = scaler.get_params()
        scalerDf = pd.DataFrame(scalerDf, index=self.trainDfIndex, columns=self.trainDfFeatureCol)
        scalerDf.insert(scalerDf.shape[1], "y", self.trainDfAnswer)

        return scalerDf, self.robustSca

    def robustTest(self, loadParams=False, loadNmlzParamsPklPath=None):
        # rs = RobustScaler()
        if loadParams:
            with open(loadNmlzParamsPklPath, 'rb') as f:
                robustSca = pickle.load(f)
                scalerDf = robustSca.fit_transform(self.testArray)
        else:
            scalerDf = self.robustSca.fit_transform(self.testArray)
        scalerDf = pd.DataFrame(scalerDf, index=self.testDfIndex, columns=self.testDfFeatureCol)
        scalerDf.insert(scalerDf.shape[1], "y", self.testDfAnswer)
        return scalerDf





