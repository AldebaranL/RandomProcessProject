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

def read_data():
    file_names=[f'data/EEGsigsimagined_subjectP1_session20170901_block{i}.mat'for i in range(1,7)]
    total_time = 100
    overlap_rate = 0.7
    eeg_data = [] #(30*block_num,sample_num(12000-16000),channel_num(24))
    lables = [] #(30*block_num)
    corr_matrixs = [] #(30*block_num,total_time,channel_num,channel_num)
    for file_name in file_names:
        data = scipy.io.loadmat(file_name)
        #print(data)
        start_time = data['prompt_start_time_marker'][0][0]
        for i,prompt_times in enumerate(data['prompt_times']):
            eeg_data.append(data['EEG_data'][int((prompt_times[1]-start_time)*2048):int((prompt_times[3]-start_time)*2048),:])
            lables.append(prompt_times[0])
            corr_matrix_list = []
            window_size = int(eeg_data[i].shape[0]/(total_time+overlap_rate-total_time*overlap_rate)-1)
            for t in range(total_time):
                window_left = int(t*window_size*(1-overlap_rate))
                sampled_data=eeg_data[i][window_left:window_left+window_size,:]
                #print(sampled_data.shape)
                corr_matrix_list.append(np.cov(sampled_data,rowvar=False))
            #print(corr_matrix_list)
            corr_matrixs.append(corr_matrix_list)
    return  eeg_data,lables,corr_matrixs

#print(len(corr_matrixs),len(corr_matrixs[0]),len(corr_matrixs[1]),corr_matrixs[1][0])

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