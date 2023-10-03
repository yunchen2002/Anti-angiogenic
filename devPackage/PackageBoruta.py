import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state
from boruta import BorutaPy
import xgboost as xgb
import lightgbm as lgb
import os


class BorutaPackage:
    def __init__(self, dataDf=None, modelName="RF", runBoruta=True, featRankPath=None):
        if modelName == "XGB":
            self.model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
        if modelName == "RF":
            self.model = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10)
        if modelName == "LGB":
            self.model = lgb.LGBMClassifier(num_boost_round=100)
        if runBoruta:
            self.returnDf = dataDf
            self.featRankPath = featRankPath
            dataDf.to_csv("../data/Boruta.csv", index=False)  # 去除pepseq欄位
            self.df = pd.read_csv("../data/Boruta.csv")
            self.modelName = modelName
            self.run()
            os.remove("../data/Boruta.csv")
        else:
            featureAll = pd.read_csv(featRankPath, index_col=[0])
            print(featureAll)
            self.feature_sort = featureAll

    def run(self):
        df = self.df
        X = df.drop(['y'], axis=1).values
        y = df['y'].values
        y = y.ravel()
        boruta_selector = BorutaPy(self.model, n_estimators='auto', verbose=2, random_state=4242)
        boruta_selector.fit(X, y)
        feature_names = np.array(df.drop(['y'], axis=1).columns)
        self.feature_ranks = list(zip(feature_names,
                                      boruta_selector.ranking_))
        featureAll = pd.DataFrame(self.feature_ranks, columns=['feature name', 'rank'])
        self.feature_sort = featureAll.sort_values(by=['rank'], ascending=True)
        self.feature_sort.to_csv(self.featRankPath)
        return self.feature_sort

    def ranks(self, number):
        feature_ranks = self.feature_sort
        dictFS = {}
        for feat in feature_ranks:
            dictFS[feat[0]] = feat[1]
        returndict = {k: v for k, v in dictFS.items() if v <= number}
        returndict.update({'y': '1'})
        key = returndict.keys()
        return key

    def numberRanks(self, number):
        feature_sort = self.feature_sort
        feature_sort = feature_sort.reset_index(drop=True)
        df = feature_sort.iloc[0:number]
        keyList = df['feature name'].tolist()
        return keyList
