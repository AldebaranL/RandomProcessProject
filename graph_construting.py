'''
用eeg数据建立邻接矩阵并画图
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 
import scipy.io
import seaborn as sns
import networkx as nx

#print(len(corr_matrixs),len(corr_matrixs[0]),len(corr_matrixs[1]),corr_matrixs[1][0])
def process_files(file_names):
    total_time = 1000
    overlap_rate = 0.6
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

def read_all_data():
    # 定义文件名
    healthy_file_names = [f'data/DATASET/EEGsigsimagined_subjectP1_session20170901_block{i}.mat' for i in range(1, 4)]
    sick_file_names_1 = [f'data/DATASET/EEGsigsactive_subjectP2_session20230927_block{i}.mat' for i in range(1, 4)]
    sick_file_names_2 = [f'data/DATASET/EEGsigsimagined_subjectP2_session20230927_block{i}.mat' for i in range(1, 7)]
    new_sick_file_names = [f'data/DATASET/EEGsigsimagined_subjectP3_session20231006_block{i}.mat' for i in range(1, 6)]
    additional_test_file_names = [f'data/DATASET/EEGsigsimagined_subjectP1_session20170901_block{i}.mat' for i in range(4, 7)]
    extra_sick_file_names = [f'data/DATASET/EEGsigsimagined_subjectP4_session20231010_block{i}.mat' for i in range(1, 6)]
    extra_test_file_names_1 = [f'data/DATASET/EEGsigsimagined_subjectP7_session20231019_block{i}.mat' for i in range(1, 4)]
    extra_test_file_names_2 = [f'data/DATASET/EEGsigsimagined_subjectP8_session20231120_block{i}.mat' for i in range(1, 4)]
    extra_train_file_names_1 = [f'data/DATASET/EEGsigsimagined_subjectP5_session20231013_block{i}.mat' for i in range(1, 4)]
    extra_train_file_names_2 = [f'data/DATASET/EEGsigsimagined_subjectP6_session20231016_block{i}.mat' for i in range(1, 4)]
    sick_file_names=sick_file_names_1#+sick_file_names_2 +new_sick_file_names+additional_test_file_names\
                    #+extra_sick_file_names+extra_test_file_names_1 +extra_test_file_names_2\
                    #+extra_train_file_names_1 +extra_train_file_names_2

    eeg_data_healthy, labels_healthy, corr_matrixs_healthy = process_files(healthy_file_names)
    eeg_data_sick, labels_sick, corr_matrixs_sick = process_files(sick_file_names)
    sick_lables = np.array([0] * len(eeg_data_healthy) + [1] * len(eeg_data_sick))
    return eeg_data_healthy+eeg_data_sick,sick_lables,corr_matrixs_healthy+corr_matrixs_sick

def show_correlation(corr_matrix):
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt=".3f",annot_kws={"size":5})
    plt.title(f'pic of correlation for file{file_name}')
    plt.xlabel('channels')
    plt.ylabel('channels')
    plt.show()
#for i in range(3):
#    show_correlation(corr_matrixs[1][i])

#channels_num = 24

def show_graph():
    G = nx.Graph()
    threshold = 10
    for i in range(channels_num):
        G.add_node(i)
        for j in range(i + 1, channels_num):
            if np.abs(corr_matrixs[0][0][i, j]) > threshold:
                G.add_edge(i, j, weight=corr_matrixs[0][0][i, j])

    if G.number_of_nodes()==0:
        print("graph is null")
    else:
        pos=nx.spring_layout(G)
        edges = G.edges(data=True)
    #print(G)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='gray', alpha=0.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{w['weight']:.2f}" for u, v, w in edges})
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title('EEG network graph')
    plt.show()

#show_graph()