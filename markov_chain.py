'''
根据图数据建立马尔可夫链
'''
import pandas as pd
import numpy as np
from graph_construting import *
import matplotlib.pyplot as plt
import pywt

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

def adj2stat_connected_components(graph_adjacency_matrix_list,channal_num,threshold = 10):
    states = set([i+1 for i in range(channal_num)])
    seq=[]
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

def adj2stat_vertex_connectivity(graph_adjacency_matrix_list,channal_num,threshold = 10):
    states = set([i for i in range(channal_num)])
    seq=[]
    for graph_adjacency_matrix in graph_adjacency_matrix_list:
        seq.append(vertex_connectivity(graph_adjacency_matrix > threshold))
    return states,seq
def vertex_connectivity(adjacency_matrix):
    n = len(adjacency_matrix)
    min_cut = n
    for row in range(n):
        # 创建一个副本，避免修改原始矩阵
        matrix_copy = np.copy(adjacency_matrix)
        # 删除一行和一列
        matrix_copy = np.delete(matrix_copy, row, axis=0)
        matrix_copy = np.delete(matrix_copy, row, axis=1)
        # 使用深度优先搜索计算连通分量数
        visited = [False] * (n - 1)
        components = 0
        for col in range(n - 1):
            if not visited[col]:
                dfs(matrix_copy, col, visited)
                components += 1
        # 更新最小割
        min_cut = min(min_cut, components)
    return min_cut
def dfs(matrix, vertex, visited):
    visited[vertex] = True
    for i in range(len(matrix)):
        if matrix[vertex][i] and not visited[i]:
            dfs(matrix, i, visited)

def wavelet_transform(eeg_data, wavelet='db4', level=3):
    features = []
    for channel_i in range(24):  # 遍历每个通道
        signal = eeg_data[:,channel_i]
        #print(len(signal))
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        #print(len(coeffs))
        # 提取统计特征
        channel_features = []
        for coeff in coeffs:
            channel_features.append(np.mean(coeff))
            channel_features.append(np.std(coeff))
            channel_features.append(np.sum(np.square(coeff)))  # 能量
        #print(len(channel_features))
        features.append(channel_features)
    features = np.hstack(features)
    print(features.shape)
    return features

def forier_transform(data):
    #data维度为[T,N]
    T,N = data.shape[0],data.shape[1]
    fft_data = np.fft.fft(data, axis=0)  # 沿时间轴进行傅里叶变换

    print(fft_data)
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
    plt.savefig('forier_transform.png')
    #plt.show()

def get_transform_feature(eeg_data):
    f = []
    print(len(eeg_data))
    for i in range(len(eeg_data)):
        f.append(wavelet_transform(eeg_data[i]))
    return np.array(f)

def get_MC_features(eeg_data,lables,corr_matrixs):
    transition_matrices = []
    channal_num = 24
    for i in range(len(eeg_data)):
        MC = []
        transition_matrix = []
        # for j in range(channal_num):
        #     states, seq = adj2stat_bynodes(corr_matrixs[i], j, channal_num)
        #     MC.append(MarkovChain(states))
        #     MC[j].fit(seq)
        #     # print(f"Sample {i}, Channel {j}: Number of states = {len(states)}")  # 打印状态数量
        #     transition_matrix.append(MC[j].trans)
        states, seq = adj2stat_connected_components(corr_matrixs[i], channal_num)
        MC_connected_components = MarkovChain(states)
        transition_matrix.append(MC_connected_components.trans)
        states, seq = adj2stat_vertex_connectivity(corr_matrixs[i], channal_num)
        MC_vertex_connectivity = MarkovChain(states)
        transition_matrix.append(MC_vertex_connectivity.trans)
        transition_matrices.append((transition_matrix))
    return np.array(transition_matrices), np.array(lables)

if __name__ == "__main__":
    eeg_data_healthy, labels_healthy, corr_matrixs_healthy = process_files2([f'data/DATASET/EEGsigsimagined_subjectP1_session20170901_block1.mat'])
    print(len(eeg_data_healthy[0][0]), labels_healthy, len(corr_matrixs_healthy))
    #f = wavelet_transform(eeg_data_healthy[0])
    #print(f)
    channal_num = 24
    states, seq = adj2stat_connected_components(corr_matrixs_healthy[0],channal_num)
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
