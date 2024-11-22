'''
根据图数据建立马尔可夫链
'''
import pandas as pd
import numpy as np
from graph_construting import read_data
import matplotlib.pyplot as plt

# T = 25
# graph_adjacency_matrix_path = "./graph_adjacency_matrix/"
# interictal_graph_adjacency_matrix_path = "./interictal_graph_adjacency_matrix/interictal_"
# preictal_graph_adjacency_matrix_path = "./preictal_graph_adjacency_matrix/preictal_"
#
# graph_adjacency_matrix = []
# for i in range(T):
#     graph_adjacency_matrix.append(pd.read_csv(graph_adjacency_matrix_path +'graph_adjacency_matrix_' + str(i+1) + ".csv", header=None))
#     #print(graph_adjacency_matrix[i].shape)

class MarkovChain:

    def __init__(self, states):
        self.states = sorted(states) # 状态集
        self.states2id = {i:j for j,i in enumerate(self.states)}
        self.id2states = {j:i for i,j in self.states2id.items()}
        self.size = len(states)
        self.pi = np.zeros((1, self.size))
        self.trans = np.zeros((self.size, self.size))

    def fit(self, seqs):
        # 初始状态的参数学习
        for seq in seqs:
            init = seq
            state_id = self.states2id[init]
            self.pi[0][state_id] += 1
        self.pi = self.pi / np.sum(self.pi)

        # 状态转移矩阵参数学习
        for i in range(len(seqs)-1):
            state1,state2 = seqs[i],seqs[i+1]
            id1 = self.states2id[state1]
            id2 = self.states2id[state2]
            self.trans[id1][id2] += 1

        # 归一化处理
        row_sums = np.sum(self.trans, axis=1, keepdims=True)
        self.trans = self.trans / np.where(row_sums == 0, 1, row_sums)  # 避免除以零

def adj2stat_bynodes(graph_adjacency_matrix_list,node_index,channal_num):
    states = set([i for i in range(channal_num+1)])
    seq=[]
    threshold = 5
    for graph_adjacency_matrix in graph_adjacency_matrix_list:
        seq.append(np.sum(graph_adjacency_matrix[node_index] > threshold))
    return states,seq

def adj2stat_eigenvalue(graph_adjacency_matrix_list,node_index,channal_num):
    states =set()
    seq=[]
    threshold = 5
    for graph_adjacency_matrix in graph_adjacency_matrix_list:
        eigenvalues, eigenvectors = np.linalg.eig(graph_adjacency_matrix)
        #seq.append(eigenvalues)
    return states,seq

def forier_transform(data):
    #data维度为[T,N]
    T,N = data.shape[0],data.shape[1]
    fft_data = np.fft.fft(data, axis=0)  # 沿时间轴进行傅里叶变换

    #print(fft_data)
    magnitude = np.abs(fft_data)
    # 可视化结果
    frequencies = np.fft.fftfreq(T, d=1 /T)  # 频率轴
    #print(frequencies)
    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(frequencies[:T//2], magnitude[:T//2, i], label=f'Channel {i + 1}')
    plt.title('Frequency Spectrum of Each Channel')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    eeg_data,lables,corr_matrixs = read_data()
    forier_transform(eeg_data[0])
    channal_num = 24
    MC=[None for i in range(channal_num)]
    for i in range(channal_num):
        states,seq = adj2stat_bynodes(corr_matrixs[0],i,channal_num)
        MC[i] = MarkovChain(states)
        MC[i].fit(seq)
        #print(seq)
        #print(states)
        #print(MC[i].pi)
        #print(MC[i].trans.shape)