import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time

UNCLASSIFIED = False
NOISE = 0

def loadDataSet(fileName, splitChar=','):
    """
    输入：文件名
    输出：数据集
    描述：从文件读入数据集
    """
    dataSet = pd.read_csv(fileName, dtype=np.float32)
    print(dataSet)
    return dataSet

def dist(a, b):
    """
    输入：向量A, 向量B
    输出：两个向量的欧式距离
    """
    return math.sqrt(np.power(a - b, 2).sum())

def eps_neighbor(a, b, eps):
    """
    输入：向量A, 向量B
    输出：是否在eps范围内
    """
    return dist(a, b) < eps

def region_query(data, pointId, eps):
    """
    输入：数据集, 查询点id, 半径大小
    输出：在eps范围内的点的id
    """
    nPoints = data.shape[1]
    seeds = []
    for i in range(nPoints):
        if eps_neighbor(data[:, pointId], data[:, i], eps):
            seeds.append(i)
    return seeds

def expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
    """
    输入：数据集, 分类结果, 待分类点id, 簇id, 半径大小, 最小点个数
    输出：能否成功分类
    这个核心函数中，会判断某个点是不同是否是核心对象，如果不是，暂时将其判断噪声点，可能会误判，但会在其他点的判断中得到纠正
    如果是核心对象，则会以此点为基础生成一个聚类，并将其周围eps距离内的点标识为同一类；在此基础上，寻找该核心对象eps距离内的其他核心对象，将另一核心对象及另一核心对象周围点划分原始
    核心对象同一类，并不判断扩展，直至找不到核心对象
    """
    seeds = region_query(data, pointId, eps)
    if len(seeds) < minPts:  # 不满足minPts条件的为噪声点（应该是非核心对象）
        clusterResult[pointId] = NOISE  # 某点不是核心对象，暂时判其为噪声点（类别用0来表示），但如果该点虽然自己不是核心对象，但在其他点判断时，如果其他点是核心对象，
        # 而它又在另一点的eps距离内，它仍然会被重新分到另一类中，因而这里不用担心被误判
        return False
    else:
        clusterResult[pointId] = clusterId  # 划分到该簇（由核心对象来代表该簇）
        for seedId in seeds:
            clusterResult[seedId] = clusterId   # 将周围的点一同划分到该簇

        while len(seeds) > 0:  # 通过判断周围的点是否为核心对象，持续扩张
            currentPoint = seeds[0]
            queryResults = region_query(data, currentPoint, eps)
            # 这里可以优化，因为如果之前已经判断为非核心对象则对应clusterResult为0,没必要再算一次，
            # 从来没判断过的其实是False,两者还是有区别的
            # 同是在seed中应该排除原始核心对象，否则在扩展时会重复算；
            # 同时利他其他核心对象扩展时，也就排除其他核心对象。修改的方法应该是region_query函数的返回值就不应该包括查询点本身。
            if len(queryResults) >= minPts:  # eps距离内的某一点是核心对象
                for i in range(len(queryResults)):
                    resultPoint = queryResults[i]
                    if clusterResult[resultPoint] == UNCLASSIFIED:
                        # 因为两个距离在eps内的核心对象会将彼此周围的点连成一片，即距离可达，因而这些点被判断为原始核心对象同一类
                        # 并且将这些点也归到原始核心对象的周围点中去，从而实现不同扩展，这是整个程序中最巧妙和关键的地方。
                        seeds.append(resultPoint)
                        clusterResult[resultPoint] = clusterId
                    elif clusterResult[resultPoint] == NOISE:  # 将另一核心对象周围的非核心对象标识为原始核心对象同一类，
                        clusterResult[resultPoint] = clusterId
            seeds = seeds[1:]   # 不断更新，已经判断过点会被移除，直至seed为空；即原始核心对象eps距离内的点都进行了是否为核心对象的判断。
        return True

def dbscan(data, eps, minPts):
    """
    输入：数据集, 半径大小, 最小点个数
    输出：分类簇id
    """
    clusterId = 1   # 类别号从1开始，用什么来标识类型本来是无所谓的，但如果从1开始，得到聚类结果的同时也就得到簇的个数。
    nPoints = data.shape[1]  # 初始每个样本的类别为False，即还未参考聚类过程，最终聚类后的结果为类别标识，即上面的clusterId
    clusterResult = [UNCLASSIFIED] * nPoints
    for pointId in range(nPoints):  # 逐个点进行类别判定
        point = data[:, pointId]
        if clusterResult[pointId] == UNCLASSIFIED:  # 如果还未被聚到某一类（虽然是逐个点判断，但只要之前已经生成了该点类别就会跳过去）
            if expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):  # 这里没有定义全局变量，expand_cluster也只返回True和False
                clusterId = clusterId + 1   # 以某一点为核心对象进行扩展，连成一片，之后就是一个类别自然就加1了。
    return clusterResult, clusterId - 1

def plotFeature(data, clusters, clusterNum):
    nPoints = data.shape[1]
    matClusters = np.mat(clusters).transpose()
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(clusterNum + 1):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[:, np.nonzero(matClusters[:, 0].A == i)]
        ax.scatter(subCluster[0, :].flatten().A[0], subCluster[1, :].flatten().A[0], c=colorSytle, s=50)

def main():
    dataSet = loadDataSet('data_sxl.csv', splitChar=',')
    dataSet = np.mat(dataSet).transpose()  # 将数据变成一列一个样本
    # print(dataSet)
    clusters, clusterNum = dbscan(dataSet, 5, 2)
    print("cluster Numbers = ", clusterNum)
    print(clusters)
    plotFeature(dataSet, clusters, clusterNum)

if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))
    plt.show()
