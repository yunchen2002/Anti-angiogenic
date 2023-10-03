from pycaret.classification import *
import os


class PycaretWrapper:
    def __init__(self):
        self.model = None
        self.modelNameList = None
        self.tunedModelList = None
        self.finalModelList = None

    def doSetup(self, trainData=None, testData=None, train_size=0.8, sessionID=None, needTrain=True):
        """

        :param needTrain:
        :param trainData:
        :param testData: testData=None 會切 trainData 當成 testData
        :param train_size: 切 trainData 的 size.
        :param sessionID: 本次實驗的種子碼, sessionId=None 會隨機帶入數字.
        :return:
        """
        if needTrain:
            self.model = setup(data=trainData,
                               test_data=testData,
                               train_size=train_size,
                               target='y',
                               feature_selection=False,
                               normalize=False,
                               silent=True,
                               preprocess=False,
                               session_id=sessionID)
            setupConfig = pull()
            setupDF = setupConfig.data

            return setupDF
        else:
            pass

    def doCompareModel(self, fold=5, n_select=None, sort='MCC', turbo=False, includeModelList=None):
        """

        :param fold: cross validation fold.
        :param includeModelList: 等於 None 的話會自動帶入 pycaret 預設的 18 個 model.
        :param n_select: 顯示 TOP N 的 model.
        :param sort: 根據哪個分數作排列.
        :param turbo: turbo=True 會減少 model 數量, 加快 compare model 速度.
        :return:
        """
        if includeModelList is None:
            includeModelList = ['rbfsvm', 'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et', 'lightgbm', 'rf', 'xgboost', 'gpc', 'mlp', 'dt', 'svm', 'qda']
        if n_select is None:
            n_select = len(includeModelList)
        defaultModelParamList = compare_models(fold=fold,
                                               n_select=n_select,
                                               sort=sort,
                                               turbo=turbo,
                                               include=includeModelList)
        scoreRank = pull()

        return defaultModelParamList, scoreRank

    def doTuneModel(self, searchLibrary='optuna', searchAlg='tpe', includeModelList=None, foldNum=5,
                    n_iter=10, early_stopping_max_iters=10, early_stopping=False,
                    customGridDict=None):
        """

        :param searchLibrary:
        :param searchAlg:
        :param includeModelList:
        :param foldNum:
        :param n_iter:
        :param early_stopping_max_iters:
        :param early_stopping:
        :param customGridDict: 將想要調整的 Ml params 範圍輸入對應的 ML model 中,
        ex: customGridDict = {'rbfsvm': {'C': [2**-5, 2**-3, 2**-1, ... , 2**15], 'gamma': [2**-15, 2**-13, 2**-11, ..., 2**3]}}
        如果 params 為 None 則 pycaret 會自己給範圍。
        ### 如果使用 scikit-learn 的 grid search 則 ML params 不能帶 None
        :return:
        """
        if includeModelList is None:
            includeModelList = ['rbfsvm', 'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et', 'lightgbm',
                                'rf', 'xgboost', 'gpc', 'mlp', 'dt', 'svm', 'qda']
        if customGridDict is None:
            customGridDict = {'rbfsvm': None, 'gbc': None, 'ridge': None, 'lr': None,
                              'catboost': None, 'lda': None, 'ada': None, 'knn': None,
                              'nb': None, 'et': None, 'lightgbm': None, 'rf': None,
                              'xgboost': None, 'gpc': None, 'mlp': None, 'dt': None,
                              'svm': None, 'qda': None}
        customGridKeysList = list(customGridDict.keys())
        customGridValuesList = list(customGridDict.values())
        self.modelNameList = includeModelList
        tunerList = []
        self.tunedModelList = []
        for (modelName, param) in zip(self.modelNameList, customGridValuesList):
            if modelName == 'lr':
                defaultModel = create_model(modelName, fold=foldNum, max_iter=2000)
            else:
                defaultModel = create_model(modelName, fold=foldNum)      # 先 create 一個原始參數的 model, 再由下一步 tune model 做 CV 及參數最佳化。
            tuneModels, tuner = tune_model(defaultModel, search_library=searchLibrary, choose_better=True, optimize='MCC',
                                           return_tuner=True, search_algorithm=searchAlg, return_train_score=True,
                                           fold=foldNum, early_stopping_max_iters=early_stopping_max_iters, n_iter=n_iter,
                                           early_stopping=early_stopping, custom_grid=param)
            self.tunedModelList.append(tuneModels)
            tunerList.append(tuner)
        return self.tunedModelList, tunerList

    def doFinalizeModel(self):
        """
        把 training 跟 self test 合在一起重新 fit
        並存在 self.finalModelList 裡面
        """
        self.finalModelList = []
        for tunedModelParam in self.tunedModelList:
            finalModel = finalize_model(tunedModelParam)
            self.finalModelList.append(finalModel)

    def doSaveModel(self, path, b_isFinalizedModel=True):
        """

        :param path:
        :param b_isFinalizedModel: 假設是 True 會 save finalized model, False 會 save tuned model. ### tunedModel 不包含 self test 作為 training
        :return:
        """
        if b_isFinalizedModel:
            for (finalModel, modelName) in zip(self.finalModelList, self.modelNameList):
                savePath = os.path.join(path, modelName+'_final')
                save_model(finalModel, savePath)
        else:
            for (tuneModel, modelName) in zip(self.tunedModelList, self.modelNameList):
                savePath = os.path.join(path, modelName+'_tuned')
                save_model(tuneModel, savePath)

    def doLoadModel(self, path, fileNameList=None, b_isFinalizedModel=True):
        """

        :param path:
        :param fileNameList:
        :param b_isFinalizedModel: 假設是 True 會 load finalized model, False 會 load tuned model. ### tunedModel 不包含 self test 作為 training
        :return:
        """
        modelList = []
        if fileNameList is None:
            fileNameList = ['rbfsvm', 'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et', 'lightgbm', 'rf', 'xgboost', 'gpc', 'mlp', 'dt', 'svm', 'qda']
        if b_isFinalizedModel:
            for fileName in fileNameList:
                loadPath = os.path.join(path, fileName+'_final')
                loadedModel = load_model(loadPath)
                resultModel = loadedModel.named_steps.trained_model
                modelList.append(resultModel)
        else:
            for fileName in fileNameList:
                loadPath = os.path.join(path, fileName+'_tuned')
                loadedModel = load_model(loadPath)
                resultModel = loadedModel.named_steps.trained_model
                modelList.append(resultModel)

        return modelList
