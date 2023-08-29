import numpy as np
import copy
import torch
'''
def sparse_matrix(similarity_matrix,p):
    length=similarity_matrix.shape[0]#一定是方阵
    N=np.zeros((length,length))
    similarity_matrix_after_sparse=np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            pNeighborsofj=pNeighbors(j,similarity_matrix,p)
            pNeighborsofi=pNeighbors(i,similarity_matrix,p)
            if ( i in pNeighborsofj ) and ( j in pNeighborsofi ):
                 N[i][j]=1
            elif (j not in pNeighborsofi ) and ( i not in pNeighborsofj ):
                 N[i][j]=0
            else: N[i][j]=0.5
    for i in range(length):
         for j in range(length):
            similarity_matrix_after_sparse[i][j]=similarity_matrix[i][j]*N[i][j]
    similarity_matrix_after_sparse=np.multiply(similarity_matrix,N)
    similarity_matrix_after_sparse=similarity_matrix_after_sparse+np.eye(length)
    return similarity_matrix_after_sparse
'''
def sparse_matrix(similarity_matrix,p):
    length=similarity_matrix.shape[0]#一定是方阵
    G=np.zeros((length,length))
    similarity_matrix_after_sparse=np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            pNeighborsofj=pNeighbors(j,similarity_matrix,p)
            pNeighborsofi=pNeighbors(i,similarity_matrix,p)
            if ( i in pNeighborsofj ) and ( j in pNeighborsofi ):
                 G[i][j]=1
            elif (j not in pNeighborsofi ) and ( i not in pNeighborsofj ):
                 G[i][j]=0
            else:
                G[i][j]=0.5
    for i in range(length):
        for j in range(length):
            similarity_matrix_after_sparse[i][j]=similarity_matrix[i][j]*G[i][j]
            similarity_matrix_after_sparse=np.multiply(similarity_matrix,G)
            similarity_matrix_after_sparse=similarity_matrix_after_sparse+np.eye(length)
            return similarity_matrix_after_sparse
def pNeighbors(node,matrix,K):#根据相似性矩阵返回K近邻
    KknownNeighbors=np.array([])
    featureSimilarity=copy.deepcopy(matrix[node])#在相似性矩阵中取出第node行
    featureSimilarity[node]=-100 #排除自身结点,使相似度为-100
    KknownNeighbors=featureSimilarity.argsort()[::-1]#按照相似度降序排序
    KknownNeighbors=KknownNeighbors[:K]#返回前K个结点的下标
    return KknownNeighbors
if __name__ == "__main__":




    Sd = np.loadtxt('dataset/drug_similarity_708_708.txt')

    St = np.loadtxt('dataset/target_similarity_1512_1512.txt')

   # Stemp=np.array([[1,0.3,0.6,0.1,0.9],
# [0.3,1,0.4,0.5,0.9],
# [0.6,0.4,1,0.8,0.1],
# [0.1,0.5,0.8,1,0.3],
# [0.9,0.9,0.1,0.3,1]
# ])
# sparse_matrix(similarity_matrix=Stemp,p=2)
   # Stemp=np.array([[1,0.3,0.9,0.9,0.9],
                   # [0.3,1,0.2,0.1,0.1],
                  #  [0.9,0.2,1,0.8,0.9],
                  #  [0.9,0.1,0.8,1,0.5],
                  #  [0.9,0.1,0.9,0.5,1]
                  #  ])
    Sd_after_sparse=sparse_matrix(similarity_matrix=Sd, p=7)

    St_after_sparse = sparse_matrix(similarity_matrix=St, p=7)
np.savetxt('dataset/drug_similarity_max_708_708_after_sparse.txt', Sd_after_sparse)
np.savetxt('dataset/target_similarity_max_1512_1512_after_sparse.txt', St_after_sparse)

print("end")
