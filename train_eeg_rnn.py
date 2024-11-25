import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os

print("Current working directory:", os.getcwd())

# 定义 MLP 模型
class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(RNNClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out
    

def read_data():
    # 定义文件名
    healthy_file_names = [f'EEGsigsimagined_subjectP1_session20170901_block{i}.mat' for i in range(1, 4)]
    sick_file_names_1 = [f'EEGsigsactive_subjectP2_session20230927_block{i}.mat' for i in range(1, 4)]
    sick_file_names_2 = [f'EEGsigsimagined_subjectP2_session20230927_block{i}.mat' for i in range(1, 7)]
    new_sick_file_names = [f'EEGsigsimagined_subjectP3_session20231006_block{i}.mat' for i in range(1, 6)]
    additional_test_file_names = [f'EEGsigsimagined_subjectP1_session20170901_block{i}.mat' for i in range(4, 7)]
    extra_sick_file_names = [f'EEGsigsimagined_subjectP4_session20231010_block{i}.mat' for i in range(1, 6)]
    extra_test_file_names_1 = [f'EEGsigsimagined_subjectP7_session20231019_block{i}.mat' for i in range(1, 7)]
    extra_test_file_names_2 = [f'EEGsigsimagined_subjectP8_session20231120_block{i}.mat' for i in range(1, 7)]
    extra_train_file_names_1 = [f'EEGsigsimagined_subjectP5_session20231013_block{i}.mat' for i in range(1, 7)]
    extra_train_file_names_2 = [f'EEGsigsimagined_subjectP6_session20231016_block{i}.mat' for i in range(1, 7)]
    total_time = 200
    overlap_rate = 0.6

    def process_files(file_names):
        eeg_data = []  # (30*block_num, sample_num(12000-16000), channel_num(24))
        labels = []  # (30*block_num)
        corr_matrixs = []  # (30*block_num, total_time, channel_num, channel_num)
        for file_name in file_names:
            if not os.path.exists(file_name):
                print(f"File not found: {file_name}")
                continue
            data = scipy.io.loadmat(file_name)
            start_time = data['prompt_start_time_marker'][0][0]
            for i, prompt_times in enumerate(data['prompt_times']):
                eeg_data.append(data['EEG_data'][int((prompt_times[1] - start_time) * 2048):int((prompt_times[3] - start_time) * 2048), :])
                labels.append(prompt_times[0])
                corr_matrix_list = []
                window_size = int(eeg_data[-1].shape[0] / (total_time + overlap_rate - total_time * overlap_rate) - 1)
                for t in range(total_time):
                    window_left = int(t * window_size * (1 - overlap_rate))
                    sampled_data = eeg_data[-1][window_left:window_left + window_size, :]
                    corr_matrix_list.append(np.cov(sampled_data, rowvar=False))
                corr_matrixs.append(corr_matrix_list)
        return eeg_data, labels, corr_matrixs

    # 分别处理各部分数据
    eeg_data_healthy, labels_healthy, corr_matrixs_healthy = process_files(healthy_file_names)
    eeg_data_sick_1, labels_sick_1, corr_matrixs_sick_1 = process_files(sick_file_names_1)
    eeg_data_sick_2, labels_sick_2, corr_matrixs_sick_2 = process_files(sick_file_names_2)
    eeg_data_new_sick, labels_new_sick, corr_matrixs_new_sick = process_files(new_sick_file_names)
    additional_test_data, additional_test_labels, additional_test_corr_matrixs = process_files(additional_test_file_names)
    extra_sick_data, extra_sick_labels, extra_sick_corr_matrixs = process_files(extra_sick_file_names)
    extra_test_data_1, extra_test_labels_1, extra_test_corr_matrixs_1 = process_files(extra_test_file_names_1)
    extra_test_data_2, extra_test_labels_2, extra_test_corr_matrixs_2 = process_files(extra_test_file_names_2)
    extra_train_data_1, extra_train_labels_1, extra_train_corr_matrixs_1 = process_files(extra_train_file_names_1)
    extra_train_data_2, extra_train_labels_2, extra_train_corr_matrixs_2 = process_files(extra_train_file_names_2)

    # 合并两种命名的患病数据
    eeg_data_sick = eeg_data_sick_1 + eeg_data_sick_2
    labels_sick = labels_sick_1 + labels_sick_2
    corr_matrixs_sick = corr_matrixs_sick_1 + corr_matrixs_sick_2

    return (eeg_data_healthy, labels_healthy, corr_matrixs_healthy, eeg_data_sick, labels_sick, corr_matrixs_sick, 
            eeg_data_new_sick, labels_new_sick, corr_matrixs_new_sick, additional_test_data, additional_test_labels, 
            additional_test_corr_matrixs, extra_sick_data, extra_sick_labels, extra_sick_corr_matrixs, 
            extra_test_data_1, extra_test_labels_1, extra_test_corr_matrixs_1, extra_test_data_2, extra_test_labels_2, 
            extra_test_corr_matrixs_2, extra_train_data_1, extra_train_labels_1, extra_train_corr_matrixs_1, 
            extra_train_data_2, extra_train_labels_2, extra_train_corr_matrixs_2)

def show_graph(corr_matrix, threshold=0.5):
    G = nx.Graph()
    num_channels = corr_matrix.shape[0]
    for i in range(num_channels):
        G.add_node(i)
        for j in range(i + 1, num_channels):
            if np.abs(corr_matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=corr_matrix[i, j])

    if G.number_of_nodes() == 0:
        print("Graph is null")
    else:
        pos = nx.spring_layout(G)
        edges = G.edges(data=True)

class MarkovChain:
    def __init__(self, states):
        self.states = sorted(states)  # 状态集
        self.states2id = {i: j for j, i in enumerate(self.states)}
        self.id2states = {j: i for i, j in self.states2id.items()}
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
        for i in range(len(seqs) - 1):
            state1, state2 = seqs[i], seqs[i + 1]
            id1 = self.states2id[state1]
            id2 = self.states2id[state2]
            self.trans[id1][id2] += 1

        # 归一化处理
        row_sums = np.sum(self.trans, axis=1, keepdims=True)
        self.trans = self.trans / np.where(row_sums == 0, 1, row_sums)  # 避免除以零

def adj2stat_bynodes(graph_adjacency_matrix_list, node_index, channal_num):
    states = set([i for i in range(channal_num + 1)])
    seq = []
    threshold = 5
    for graph_adjacency_matrix in graph_adjacency_matrix_list:
        seq.append(np.sum(graph_adjacency_matrix[node_index] > threshold))
    return states, seq

def generate_transition_matrix_and_adj_features(eeg_data, corr_matrixs):
    channal_num = 24
    transition_matrices = []
    adj_features = []
    for i in range(len(eeg_data)):
        MC = [None for _ in range(channal_num)]
        adj_matrix = np.zeros((channal_num, channal_num))
        for j in range(channal_num):
            states, seq = adj2stat_bynodes(corr_matrixs[i], j, channal_num)
            MC[j] = MarkovChain(states)
            MC[j].fit(seq)
            adj_matrix[j] = np.sum(corr_matrixs[i][j] > 0, axis=0)  # 生成邻接矩阵特征
            #print(f"Sample {i}, Channel {j}: Number of states = {len(states)}")  # 打印状态数量
        transition_matrix = np.array([mc.trans for mc in MC])
        transition_matrices.append(transition_matrix)
        adj_features.append(adj_matrix.flatten())  # 展平邻接矩阵特征
    return np.array(transition_matrices), np.array(adj_features)


def train(X_train, y_train, X_test, y_test):
    # 超参数设置
    input_dim = X_train.shape[2]  # 输入特征维度
    hidden_dim = 128  # 隐藏层单元数
    output_dim = 2  # 输出类别数（健康和患病）
    num_layers = 2  # RNN 层数

    # 模型、损失函数和优化器
    model = RNNClassifier(input_dim, hidden_dim, output_dim, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 60
    for epoch in range(num_epochs):
        model.train()
        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.long)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 测试模型
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).argmax(axis=1)
        accuracy = (predictions == y_test_tensor).float().mean()
        print(f'Test Accuracy: {accuracy:.4f}')
        print("\nClassification Report:\n", classification_report(y_test, predictions.numpy(), target_names=['Healthy', 'Sick'], labels=[0, 1]))


if __name__ == "__main__":
    # 读取数据
    (eeg_data_healthy, labels_healthy, corr_matrixs_healthy, eeg_data_sick, labels_sick, corr_matrixs_sick, 
     eeg_data_new_sick, labels_new_sick, corr_matrixs_new_sick, additional_test_data, additional_test_labels, 
     additional_test_corr_matrixs, extra_sick_data, extra_sick_labels, extra_sick_corr_matrixs, 
     extra_test_data_1, extra_test_labels_1, extra_test_corr_matrixs_1, extra_test_data_2, extra_test_labels_2, 
     extra_test_corr_matrixs_2, extra_train_data_1, extra_train_labels_1, extra_train_corr_matrixs_1, 
     extra_train_data_2, extra_train_labels_2, extra_train_corr_matrixs_2) = read_data()
    
    # 检查读取的数据
    print(f"Number of healthy samples: {len(eeg_data_healthy)}")
    print(f"Number of sick samples: {len(eeg_data_sick)}")
    print(f"Number of new sick samples: {len(eeg_data_new_sick)}")
    print(f"Number of additional test samples: {len(additional_test_data)}")
    print(f"Number of extra sick samples: {len(extra_sick_data)}")
    print(f"Number of extra test samples 1: {len(extra_test_data_1)}")
    print(f"Number of extra test samples 2: {len(extra_test_data_2)}")
    print(f"Number of extra train samples 1: {len(extra_train_data_1)}")
    print(f"Number of extra train samples 2: {len(extra_train_data_2)}")
    
    # 生成状态转移概率矩阵和邻接矩阵特征
    if len(eeg_data_healthy) > 0:
        transition_matrices_healthy, adj_features_healthy = generate_transition_matrix_and_adj_features(eeg_data_healthy, corr_matrixs_healthy)
    if len(eeg_data_sick) > 0:
        transition_matrices_sick, adj_features_sick = generate_transition_matrix_and_adj_features(eeg_data_sick, corr_matrixs_sick)
    if len(eeg_data_new_sick) > 0:
        transition_matrices_new_sick, adj_features_new_sick = generate_transition_matrix_and_adj_features(eeg_data_new_sick, corr_matrixs_new_sick)
    if len(additional_test_data) > 0:
        transition_matrices_additional_test, adj_features_additional_test = generate_transition_matrix_and_adj_features(additional_test_data, additional_test_corr_matrixs)
    if len(extra_sick_data) > 0:
        transition_matrices_extra_sick, adj_features_extra_sick = generate_transition_matrix_and_adj_features(extra_sick_data, extra_sick_corr_matrixs)
    if len(extra_test_data_1) > 0:
        transition_matrices_extra_test_1, adj_features_extra_test_1 = generate_transition_matrix_and_adj_features(extra_test_data_1, extra_test_corr_matrixs_1)
    if len(extra_test_data_2) > 0:
        transition_matrices_extra_test_2, adj_features_extra_test_2 = generate_transition_matrix_and_adj_features(extra_test_data_2, extra_test_corr_matrixs_2)
    if len(extra_train_data_1) > 0:
        transition_matrices_extra_train_1, adj_features_extra_train_1 = generate_transition_matrix_and_adj_features(extra_train_data_1, extra_train_corr_matrixs_1)
    if len(extra_train_data_2) > 0:
        transition_matrices_extra_train_2, adj_features_extra_train_2 = generate_transition_matrix_and_adj_features(extra_train_data_2, extra_train_corr_matrixs_2)
    
    # 确保所有矩阵都存在
    if (len(eeg_data_healthy) == 0 or len(eeg_data_sick) == 0 or len(eeg_data_new_sick) == 0 or 
        len(additional_test_data) == 0 or len(extra_sick_data) == 0 or len(extra_test_data_1) == 0 or 
        len(extra_test_data_2) == 0 or len(extra_train_data_1) == 0 or len(extra_train_data_2) == 0):
        print("Error: Not all data sets are available.")
        exit(1)
    
    # 将状态转移概率矩阵和邻接矩阵特征展平为特征向量
    num_samples_healthy, num_channels, num_states, _ = transition_matrices_healthy.shape
    num_samples_sick = transition_matrices_sick.shape[0]
    num_samples_new_sick = transition_matrices_new_sick.shape[0]
    num_samples_additional_test = transition_matrices_additional_test.shape[0]
    num_samples_extra_sick = transition_matrices_extra_sick.shape[0]
    num_samples_extra_test_1 = transition_matrices_extra_test_1.shape[0]
    num_samples_extra_test_2 = transition_matrices_extra_test_2.shape[0]
    num_samples_extra_train_1 = transition_matrices_extra_train_1.shape[0]
    num_samples_extra_train_2 = transition_matrices_extra_train_2.shape[0]
    
    X_healthy = np.hstack((transition_matrices_healthy.reshape(num_samples_healthy, -1), adj_features_healthy))
    X_sick = np.hstack((transition_matrices_sick.reshape(num_samples_sick, -1), adj_features_sick))
    X_new_sick = np.hstack((transition_matrices_new_sick.reshape(num_samples_new_sick, -1), adj_features_new_sick))
    X_additional_test = np.hstack((transition_matrices_additional_test.reshape(num_samples_additional_test, -1), adj_features_additional_test))
    X_extra_sick = np.hstack((transition_matrices_extra_sick.reshape(num_samples_extra_sick, -1), adj_features_extra_sick))
    X_extra_test_1 = np.hstack((transition_matrices_extra_test_1.reshape(num_samples_extra_test_1, -1), adj_features_extra_test_1))
    X_extra_test_2 = np.hstack((transition_matrices_extra_test_2.reshape(num_samples_extra_test_2, -1), adj_features_extra_test_2))
    X_extra_train_1 = np.hstack((transition_matrices_extra_train_1.reshape(num_samples_extra_train_1, -1), adj_features_extra_train_1))
    X_extra_train_2 = np.hstack((transition_matrices_extra_train_2.reshape(num_samples_extra_train_2, -1), adj_features_extra_train_2))
    
    # 合并健康和患病数据
    X_train = np.vstack((X_healthy, X_sick, X_extra_sick, X_extra_train_1, X_extra_train_2))
    y_train = np.array([0] * num_samples_healthy + [1] * (num_samples_sick + num_samples_extra_sick + num_samples_extra_train_1 + num_samples_extra_train_2))
    
    # 数据预处理
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # 确保测试集包含健康样本和患病样本
    num_test_healthy = min(num_samples_healthy, num_samples_new_sick // 2)
    num_test_sick = num_samples_new_sick - num_test_healthy
    
    X_test = np.vstack((X_healthy[:num_test_healthy], X_new_sick[:num_test_sick], X_additional_test, X_extra_test_1, X_extra_test_2))
    y_test = np.array([0] * num_test_healthy + [1] * (num_test_sick + num_samples_additional_test + num_samples_extra_test_1 + num_samples_extra_test_2))
    
    X_test = scaler.transform(X_test)
    
    # 将数据转换为RNN输入格式
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    # 训练和测试模型
    train(X_train, y_train, X_test, y_test)