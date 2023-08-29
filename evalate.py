import numpy as np
from pylab import *
from sklearn.metrics import *
from sklearn.metrics import f1_score
from my_function import *
from tqdm import *
from matplotlib import pyplot as plt

from sklearn import metrics
FOLD = 5


def null_list(num):
    lis = []
    for i in range(num): lis.append([])
    return lis


def equal_len_list(a):      # 按比例采样
    row_len = []
    for i in a:
        row_len.append(len(i))
    min_len = min(row_len)
    equal_len_A = []
    for i in a:
        tem_list = []
        multi = len(i)/min_len
        for j in range(min_len):
            tem_list.append(i[int(j*multi)])
        equal_len_A.append(tem_list)
    return equal_len_A


AUC, AUPR, F1, ACC, RECALL = [], [], [], [], []
for f in range(FOLD):
    DTI = np.loadtxt('dataset/mat_drug_protein.txt')
    y_pred1 = np.loadtxt('result/fold_' + str(f) + '_pre.txt')
    y_pred = np.zeros((y_pred1.shape[0]))
    for i in range(y_pred.shape[0]):
        y_pred[i] = y_pred1[i, 1] / (y_pred1[i, 0] + y_pred1[i, 1])

    y_true = np.loadtxt('result/fold_' + str(f) + '_label.txt')
    s_1 = y_pred[0: 384]
    l_1 = np.ones((384,))
    # s_0 = y_pred[384:]

    s_0 = []
    for i in range(384, y_pred.shape[0]):
        if y_pred[i] < 0.5:
            s_0.append(y_pred[i])
    s_0_sample = np.array(s_0[0: 384])
    l_0_sample = np.zeros((384, ))
    s = np.hstack((s_0_sample, s_1))
    l = np.hstack((l_0_sample, l_1))

    fpr, tpr, th = roc_curve(l, s)
    # print("FOLD:" + str(f) + " AUC: " + str(auc(fpr, tpr)))
    AUC.append(auc(fpr, tpr))
    # np.savetxt('result/FOLD'+str(f)+'recall.txt',tpr,fmt='%6f')

    pre, recall, th = precision_recall_curve(l, s)
    AUPR.append(auc(recall, pre))
    # print("FOLD:" + str(f) + " AUPR: " + str(auc(recall, pre)))
    # fig = plt.figure(figsize=(4, 4), dpi=300)
    # x = fpr
    # y = tpr
    # plt.plot(x, y, lw=4, ls='-', c='b', alpha=0.1)
    # plt.plot()
    # plt.show()
    # fig.savefig("画布")
    # fig = plt.figure(figsize=(4, 4), dpi=300)
    # x=recall
    # y=pre
    # plt.plot(x,y,lw=4,ls='-',c='b',alpha=0.1)
    # plt.plot()
    # plt.show()
    # fig.savefig("画布_")
    for i in range(len(s)):
        if s[i] <= 0.5:
            s[i] = 0
        else:
            s[i] = 1
    #F1 = f1_score(l, s, average='macro')
    F1.append(f1_score(l, s, average='macro'))
    ACC.append(accuracy_score(s, l))

    print("FOLD:" + str(f) + " F1: " + str(F1))
    acc = accuracy_score(s, l)
    print("FOLD:" + str(f) + " acc: " + str(acc))
    Recall= metrics.recall_score(l, s)

    #Recall = recall_score(s, l)

    #Recall.append(recall_score(s, l))
    print("FOLD:" + str(f) + " Recall: " + str(Recall))
print(' AUC: ' + str(np.average(AUC)) + ' AUPR: ' + str(np.average(AUPR)) + ' F1: ' + str(np.average(F1)) + ' ACC: ' + str(np.average(ACC)))