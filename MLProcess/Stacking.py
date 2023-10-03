from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Predict import Predict
from MLProcess.Scoring import Scoring
from MLProcess.DrawPlot import DrawPlot
from MLProcess.StackingModelSelector_HC import StackingModelSelector_HC
import pandas as pd


class Stacking:
    def __init__(self, loadModelPath, modelNameList, trainDataPath, testDataPath, clusterNumList):
        self.loadModelPath = loadModelPath
        self.modelNameList = modelNameList
        dataTrain = pd.read_csv(trainDataPath, index_col=[0])
        self.dataTrain_X = dataTrain.drop(['y'], axis=1)
        self.dataTrain_y = dataTrain[['y']]
        dataTest = pd.read_csv(testDataPath, index_col=[0])
        self.dataTest_X = dataTest.drop(['y'], axis=1)
        self.dataTest_y = dataTest[['y']]
        self.clusterNumList = clusterNumList
        self.selfTestProbVectorList = None
        self.selfTestScoreDf = None
        self.stkModelList = []

    def genSelfTestResult(self, drawPlot=False):
        selfTestPycObj = PycaretWrapper()
        loadModelList = selfTestPycObj.doLoadModel(self.loadModelPath, fileNameList=self.modelNameList, b_isFinalizedModel=False)
        if loadModelList[0].__class__.__name__ == 'CatBoostClassifier':
            random_state = loadModelList[0].random_seed_
        else:
            random_state = loadModelList[0].random_state
        X_train, X_test, y_train, y_test = train_test_split(self.dataTrain_X, self.dataTrain_y, test_size=0.2, random_state=random_state)
        selfTestPredObj = Predict(dataX=X_test, modelList=loadModelList)
        selfTestPredVectorList, self.selfTestProbVectorList = selfTestPredObj.doPredict()
        selfTestScoreObj = Scoring(predVectorList=selfTestPredVectorList, probVectorList=self.selfTestProbVectorList,
                                   answerDf=y_test, modelNameList=self.modelNameList)
        self.selfTestScoreDf = selfTestScoreObj.doScoring(sortColumn=None)
        if drawPlot:
            drawSelfTestObj = DrawPlot(answerDf=y_test, modelList=loadModelList, modelNameList=self.modelNameList, predArrList=selfTestPredVectorList, probArrList=self.selfTestProbVectorList)
            drawSelfTestObj.drawROC()

        return self.selfTestScoreDf

    def genStkModel(self, selfTestScoreLabel='auc', final_estimator=None, drawPlot=False, metric='euclidean', linkageType='ward'):
        # selfTestScore['auc']命名
        if final_estimator is None:
            final_estimator = LogisticRegression()
            print('!!!Warning!!!')
            print('Because you do not input final_estimator,')
            print('So the final_estimator is using default LogisticRegression.')
        probDf = pd.DataFrame(self.selfTestProbVectorList, index=self.modelNameList)
        stackHcObj = StackingModelSelector_HC(data=probDf, scoreDf=self.selfTestScoreDf[selfTestScoreLabel])
        for clusterNum in self.clusterNumList:
            estimatorList = []
            clustResultDf, seletedModelDf = stackHcObj.doClustering(clusterNum=clusterNum, metric=metric, linkageType=linkageType)
            seletedModelNameList = seletedModelDf[0].values.tolist()
            stkPycObj = PycaretWrapper()
            seletedModelList = stkPycObj.doLoadModel(self.loadModelPath, fileNameList=seletedModelNameList,
                                                     b_isFinalizedModel=False)
            for (modelName, model) in zip(seletedModelNameList, seletedModelList):
                estimatorList.append((modelName, model))
            stkObj = StackingClassifier(estimators=estimatorList,
                                        final_estimator=final_estimator,
                                        cv=5,
                                        stack_method='auto',
                                        n_jobs=None,
                                        passthrough=False,
                                        verbose=0)
            stkClassifier = stkObj.fit(self.dataTrain_X, self.dataTrain_y)
            self.stkModelList.append(stkClassifier)
            print('Stacking model with ' + str(clusterNum) + ' estimators: ' + str(seletedModelNameList))
        if drawPlot:
            stackHcObj.drawDendrogram()

        return self.stkModelList

    def stkModelPredictScoring(self, scoreCsvPath, drawPlot=False, b_isBinary=True, probCutOff=None):
        stkPredObj = Predict(dataX=self.dataTest_X, modelList=self.stkModelList)
        stkPredVectorList, stkProbVectorList = stkPredObj.doPredict()
        stkWordlist = ['stacking_'] * len(self.clusterNumList)
        modifiedList = list(map(lambda x, y: x + '_' + str(y), stkWordlist, self.clusterNumList))
        if b_isBinary:
            stkScoreObj = Scoring(predVectorList=stkPredVectorList, probVectorList=stkProbVectorList,
                                  answerDf=self.dataTest_y, modelNameList=modifiedList)
            stkScoreDf = stkScoreObj.doScoring(b_optimizedMcc=False, path=scoreCsvPath,
                                               sortColumn='mcc')
            if drawPlot:
                drawStkObj = DrawPlot(answerDf=self.dataTest_y, modelList=self.stkModelList, modelNameList=modifiedList,
                                      predArrList=stkPredVectorList, probArrList=stkProbVectorList)
                drawStkObj.drawROC()
        else:
            stkScoreObj = Scoring(predVectorList=stkPredVectorList, probVectorList=stkProbVectorList,
                                  answerDf=self.dataTest_y,
                                  modelNameList=modifiedList)
            stkScoreObj.optimizeMcc(cutOffList=probCutOff)
            stkScoreDf = stkScoreObj.doScoring(b_optimizedMcc=True,  path=scoreCsvPath,
                                               sortColumn='mcc')
            if drawPlot:
                drawStkObj = DrawPlot(answerDf=self.dataTest_y, modelList=self.stkModelList, modelNameList=modifiedList,
                                      predArrList=None, probArrList=stkProbVectorList)
                drawStkObj.drawROC()
        return stkScoreDf
