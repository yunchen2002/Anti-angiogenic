import sys


class Predict:
    def __init__(self, dataX, modelList):
        self.dataX = dataX
        self.predVectorList = []
        self.probVectorList = []
        self.modelList = modelList

    def doPredict(self):
        """
        :return: predVectorList: 一個 preditor 會產生一組 (0 or 1) ndarray, 多個 ndarray 會形成一個 list, 會同時算分。
                 probVectorList: 一個 preditor 會產生一組 probability ndarray, 多個 ndarray 會形成一個 list, 會同時算分。
        """

        # 底下 Ridge 和 SGD 無法預測 probability 所以用 (0 or 1) 的 predict vector 取代
        # 其餘 predictor 產生 proba vector
        for model in self.modelList:
            predVector = model.predict(self.dataX)
            self.predVectorList.append(predVector)
            if model.__class__.__name__ == 'RidgeClassifier':
                probVector = model.predict(self.dataX)
                self.probVectorList.append(probVector)
            elif model.__class__.__name__ == 'SGDClassifier':
                probVector = model.predict(self.dataX)
                self.probVectorList.append(probVector)
            else:
                probVector = model.predict_proba(self.dataX)
                self.probVectorList.append(probVector[:, 1])

        return self.predVectorList, self.probVectorList
