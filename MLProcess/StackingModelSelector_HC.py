import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt


class StackingModelSelector_HC:
    def __init__(self, data, scoreDf):
        """

        :param data:
        :param scoreDf: 只有單一分數 (single column)的 DataFrame
        """
        self.clusObj = None
        self.data = data
        self.scoreDf = scoreDf

    def doClustering(self, metric='euclidean', linkageType='ward', clusterNum=5):
        """
        :param metric: 有 ‘braycurtis’, ‘canberra’, ‘chebyshev’,
                       ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’,
                       ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘kulczynski1’,
                       ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’,
                       ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’,
                       ‘sqeuclidean’, ‘yule’ 等可以選擇。
        :param linkageType: 有 single, complete, average, weighted, centroid, median, ward 等可以選擇。
        :param clusterNum:
        :return:
        """
        scoreList = self.scoreDf.tolist()
        self.clusObj = linkage(self.data, metric=metric, method=linkageType)
        clusterArr = fcluster(self.clusObj, clusterNum, criterion='maxclust')
        clusterDf = pd.DataFrame(clusterArr, index=self.scoreDf.index.to_list(), columns=['cluster'])
        clusterDf['score'] = scoreList
        times = 1
        bestModelList = []
        while times <= clusterNum:
            selectModelDf = clusterDf.loc[clusterDf['cluster'] == times]
            selectModelDf.columns = ['cluster', 'score']
            bestModelList.append(selectModelDf['score'].idxmax())
            times += 1
        bestModelDf = pd.DataFrame(bestModelList)           #存放每個cluster最好的model
        bestModelDf.index = bestModelDf.index + 1

        return clusterDf, bestModelDf

    def drawDendrogram(self, figSize=(16, 9), dpi=300, save=False, saveLoc="dendrogram.png"):
        plt.figure(figsize=figSize, dpi=dpi)
        dendrogram(self.clusObj,
                   labels=self.data.index.tolist())
        plt.title('Hierarchical Clustering')
        if save:
            plt.savefig(saveLoc)
        plt.show()
