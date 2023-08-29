
# 1.读数据，
# 2.五倍交叉验证（对原始数据的一个划分，测试集训练集），
# 3.对数据预处理（WKNKN，归一化）
# 4.丢到神经网络里，做训练和预测
# 5.对预测结果，做一个准确性的评估
import random

import numpy as np
# 5000个元素是0 ->  第1513个元素 原矩阵里第2行第1列的那个元素 -> 1513/1512=1+1.......1
#
# 行： 5000/1512=3+1......464



def KNearestKnownNeighbors(s, k):
    num = s.shape[0]
    s = s - np.eye(num)

    s_idx = np.argsort(-s)
    s_idx_k = s_idx[0:num, 0:k]

    return s_idx_k


def get_weight(knn, s, a):
    weight = np.zeros(knn.shape)
    r = weight.shape[0]
    c = weight.shape[1]
    for i in range(r):
        for j in range(c):
            neighbor = knn[i, j]
            s_neighbor = s[i, neighbor]
            weight[i, j] = a ** j * s_neighbor

    return weight


def WKNKN(knn, w, y):
    fenmu = np.sum(w, axis=1)
    Y1 = np.zeros(y.shape)
    for i in range(knn.shape[0]):
        for j in range(knn.shape[1]):
            d_neighbor = knn[i, j]
            Y1[i, :] = Y1[i, :] + w[i, j] * y[d_neighbor, :]
        Y1[i, :] = Y1[i, :] / fenmu[i]

    return Y1


     #return y

if __name__ == '__main__':
    Y = np.loadtxt('dataset/mat_drug_protein.txt')  # 708行1512列  1070496维
    sd = np.loadtxt('dataset/Similarity_Matrix_Drugs.txt')
    sp = np.loadtxt('dataset/Similarity_Matrix_Proteins.txt')

    d_knn = KNearestKnownNeighbors(sd, 15)
    p_knn = KNearestKnownNeighbors(sp, 15)
    wd = get_weight(d_knn, sd, 0.7)
    wp = get_weight(p_knn, sp, 0.7)
    YD = WKNKN(d_knn, wd, Y)
    YP = WKNKN(p_knn, wp, Y.T).T
    YDP = (YD + YP) / 2
    Y2 = np.zeros(Y.shape)
    for i in range(Y2.shape[0]):
        for j in range(Y2.shape[1]):
            Y2[i, j] = max(Y[i, j], YDP[i,j])
Y3 = Y2.flatten()            # 1070496维
list_1 = []
list_0 = []
for i in range(len(Y3.tolist())):
    if Y3[i] == 1:
        list_1.append(i)
    elif Y3[i] == 0 :
        list_0.append(i)

random.shuffle(list_1)
random.shuffle(list_0)
num1 = len(list_1)
num0 = len(list_0)
group_size_1 = int(num1 / 5)    # 1923个1，分五组，是分不干净的，
group_size_0 = int(num0 / 5)

# 5， 7， 9 ， 13，  16是1
#
# 1 3 15 14 20 22 25 27 29 31是0
#
# 5 1 3
# 7 15 14
# 9 20 22
# 13 25 27
# 16 2 31

group_1 = np.array(list_1[0: 5*group_size_1]).reshape([5, group_size_1])
group_0 = np.array(list_0[0: 5*group_size_0]).reshape([5, group_size_0])
np.savetxt('dataset/Y_Py.txt', Y2, fmt='%6f')
np.savetxt('result1/index_1.txt', group_1, fmt='%d')
np.savetxt('result1/index_0.txt', group_0, fmt='%d')
group_10 = np.hstack((group_1, group_0))
print('1')