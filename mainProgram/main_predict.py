from userPackage.Package_Encode import EncodeAllFeatures
from userPackage.LoadDataset import LoadDataset
from userPackage.Package_CheckData import CheckData
import pandas as pd
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Scoring import Scoring
from MLProcess.Predict import Predict
from MLProcess.DrawPlot import DrawPlot

ldObj = LoadDataset()
indpNegPath = '../data/HemoPI_1_pos_val20%.fasta'
indpPosPath = '../data/HemoPI_1_neg_val20%.fasta'
# indpUNKPath = ''

# output predOutputPath
indpNegSeqDict = ldObj.readFasta(indpNegPath, minSeqLength=5)
indpPosSeqDict = ldObj.readFasta(indpPosPath, minSeqLength=5)
# indpUNKSeqDict = ldObj.readFasta()

testDict = {0: indpNegSeqDict, 1: indpPosSeqDict, -1: None}
# testDict = {0: None, 1: None, -1: indpUNKSeqDict}

featureNum = 90
mlDataPath = "../data/mlData/"  # 內含 data 檔案 ex : train_F390.csv, boruta 檔案 ex :Boruta-featRank-RF.csv
paramPath = "../data/param/"  # 內含檔案: featureTypeDict.pkl, normalize.pkl
finalModelPath = "../data/finalModel"  # train 好且 finalize 的 model 內含檔案 ex: lr_final.pkl
mlScorePath = "../data/mlScore/"
normalizeMethod = 'standard'
dataName = 'HemoPi_1'

# ==================================================================================
encodeObj = EncodeAllFeatures()
encodeObj.dataEncodeSetup(saveFeatureDict=None,  # normalization 前傳出來
                          savePklPath=None,
                          loadPklPath=paramPath + f'{dataName}_featureTypeDict.pkl',
                          b_loadPlk=True)
encodeIndpDf = encodeObj.dataEncodeOutPut(dataDict=testDict)
indpNmlzDf = encodeObj.dataNormalization(encodIndpDf=encodeIndpDf,
                                         normalization=normalizeMethod,
                                         loadNmlzScalerPklPath=paramPath + f'{dataName}_{normalizeMethod}Scaler.pkl',  # 檔名修改
                                         b_loadPkl=True)
brtObj = encodeObj.dataBoruta(trainDf=None, runBoruta=False,
                              featRankPath=mlDataPath + 'Boruta-featureRank-XGB.csv',
                              skipFeatureList=[])
encodeObj.dataDecidedFeatureNum(featureNum=featureNum, saveCsvPath=mlDataPath, indpDf=indpNmlzDf, brtObj=brtObj)
print("encoding feature done")

# =========================================================================================================================================================
modelNameList = ['lightgbm', 'rbfsvm', 'gbc', 'ridge', 'lr', 'lda', 'ada', 'knn', 'nb', 'et', 'rf', 'xgboost', 'gpc',
                 'mlp', 'dt', 'svm', 'qda']  # 發表論文時只需留 1-2 個最好的
pycObj = PycaretWrapper()
setupDf = pycObj.doSetup(needTrain=False)
dataIndpDf = pd.read_csv(mlDataPath + "/indp_F" + str(featureNum) + ".csv", index_col=[0])
dataIndp_X = dataIndpDf.drop(["y"], axis=1)
dataIndp_y = dataIndpDf[["y"]]
finalModelList = pycObj.doLoadModel(path=finalModelPath,                      # finalize 完最佳化的 model
                                    fileNameList=modelNameList, b_isFinalizedModel=True)

predObjIndp = Predict(dataX=dataIndp_X, modelList=finalModelList)
predVectorListIndp, probVectorListIndp = predObjIndp.doPredict()
predVectorDf = pd.DataFrame(predVectorListIndp, index=modelNameList, columns=dataIndpDf.index).T
probVectorDf = pd.DataFrame(probVectorListIndp, index=modelNameList, columns=dataIndpDf.index).T
predVectorDf.to_csv(mlScorePath + 'predVector.csv')
probVectorDf.to_csv(mlScorePath + 'probVector.csv')


# =======================================================================================================================
# 包成 def scoring


def doScoring(predVectorListIndp, probVectorListIndp, dataIndp_y, modelNameList, finalModelList):
    scoreObjIndp = Scoring(predVectorList=predVectorListIndp, probVectorList=probVectorListIndp,
                           answerDf=dataIndp_y, modelNameList=modelNameList)
    scoreObjIndp.optimizeMcc(cutOffList=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], method='acc')
    scoreDfIndp = scoreObjIndp.doScoring(b_optimizedMcc=True, sortColumn='accuracy',
                                         path=mlScorePath + 'singleModelScore.csv')

    drawObj = DrawPlot(answerDf=dataIndp_y, modelList=finalModelList, modelNameList=modelNameList,
                       predArrList=predVectorListIndp, probArrList=probVectorListIndp)
    aucDf = drawObj.drawROC(colorList=None, title=False, titleName='Receiver Operating Characteristic', setDpi=True,
                            legendSize=11, labelSize=20, save=True, saveLoc=mlScorePath + 'singleModelIndeROC.png',
                            show=True,
                            dpi=300, figSize=(12, 9), topNum=5)

    return scoreDfIndp


scoreDfIndp = doScoring(predVectorListIndp, probVectorListIndp, dataIndp_y, modelNameList, finalModelList)
