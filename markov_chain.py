'''
根据图数据建立马尔可夫链
'''
import pandas as pd
import numpy as np
from graph_construting import *
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
    states = set([i for i in range(channal_num)])
    seq=[]
    threshold = 5
    for graph_adjacency_matrix in graph_adjacency_matrix_list:
        seq.append(np.sum(graph_adjacency_matrix[node_index] > threshold))
    return states,seq

def adj2stat_eigenvalue(graph_adjacency_matrix_list,node_index,channal_num):
    #TODO
    states =set()
    seq=[]
    threshold = 5
    for graph_adjacency_matrix in graph_adjacency_matrix_list:
        eigenvalues, eigenvectors = np.linalg.eig(graph_adjacency_matrix)
        #seq.append(eigenvalues)
    return states,seq

def adj2stat_connected_components(graph_adjacency_matrix_list,channal_num):
    states = set([i+1 for i in range(channal_num)])
    seq=[]
    threshold = 5
    for graph_adjacency_matrix in graph_adjacency_matrix_list:
        seq.append(count_connected_components(graph_adjacency_matrix > threshold))
    return states,seq

def count_connected_components(adj_matrix):
    n = len(adj_matrix)
    visited = [False] * n
    def dfs(node):
        # 标记当前节点为已访问
        visited[node] = True
        # 访问所有相邻的未访问节点
        for neighbor in range(n):
            if adj_matrix[node][neighbor] != 0 and not visited[neighbor]:
                dfs(neighbor)
    connected_components = 0
    for i in range(n):
        if not visited[i]:
            # 从未访问的节点开始新的 DFS，找到一个连通分支
            dfs(i)
            connected_components += 1
    return connected_components

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

def get_features(eeg_data,lables,corr_matrixs):
    transition_matrices = []
    channal_num = 24
    for i in range(len(eeg_data)):
        MC = []
        transition_matrix = []
        for j in range(channal_num):
            states, seq = adj2stat_bynodes(corr_matrixs[i], j, channal_num)
            MC.append(MarkovChain(states))
            MC[j].fit(seq)
            # print(f"Sample {i}, Channel {j}: Number of states = {len(states)}")  # 打印状态数量
            transition_matrix.append(MC[j].trans)
        states, seq = adj2stat_connected_components(corr_matrixs[0], channal_num)
        MC_connected_components = MarkovChain(states)
        transition_matrix.append(MC_connected_components.trans)
        transition_matrices.append((transition_matrix))
    return np.array(transition_matrices), np.array(lables)

if __name__ == "__main__":
    eeg_data,lables,corr_matrixs = read_data()
    forier_transform(eeg_data[0])
    channal_num = 24

    states, seq = adj2stat_connected_components(corr_matrixs[0],channal_num)
    MC = MarkovChain(states)
    MC.fit(seq)
    print(seq)
    print(states)
    print(MC.pi)
    print(MC.trans)

    # MC=[None for i in range(channal_num)]
    #for i in range(channal_num):
        #states,seq = adj2stat_bynodes(corr_matrixs[0],i,channal_num)
        #MC[i] = MarkovChain(states)
        #MC[i].fit(seq)
