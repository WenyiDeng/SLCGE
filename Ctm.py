import numpy as np
from sklearn.metrics import jaccard_score
from tqdm import tqdm
# 读取数据
matrix1 = np.loadtxt('dataset/mat_drug_disease.txt')
matrix2 = np.loadtxt('dataset/mat_protein_disease.txt')
Y = np.loadtxt('dataset/mat_drug_protein.txt')

def jaccard_similarity(matrix1, matrix2):
    intersection = np.dot(matrix1, matrix2.T)
    sum_matrix1 = np.sum(matrix1, axis=1).reshape(-1, 1)
    sum_matrix2 = np.sum(matrix2, axis=1).reshape(-1, 1)
    union = sum_matrix1 + sum_matrix2.T - intersection
    jaccard_matrix = intersection / union
    return jaccard_matrix

# 计算杰卡德相似性(下面这种方式也可以计算，但速度很慢)
# result_matrix = np.zeros((matrix1.shape[0], matrix2.shape[0]))
#
# for i in tqdm(range(matrix1.shape[0])):
#     for j in range(matrix2.shape[0]):
#         result_matrix[i][j] = jaccard_score(matrix1[i], matrix2[j])

result_matrix = jaccard_similarity(matrix1, matrix2)
 x1= np.maximum(result_matrix,Y)



# 保存结果
np.savetxt('dataset/DTI_708_1512_MAX_DISCRETIZE.txt', x1, fmt='%.5f', delimiter=' ')

# 输出结果
print(np.round(result_matrix, 5))
print(result_matrix.shape)
