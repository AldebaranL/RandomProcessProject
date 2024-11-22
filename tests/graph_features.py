import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io

current_directory=os.getcwd()
#print(f"current director:{current_directory}")
file_names=[f'EEGsigsimagined_subjectP1_session20170901_block{i}.mat'for i in range(1,7)]

all_channels=[]

for file_name in file_names:
    data=scipy.io.loadmat(file_name)
    key_data=data['EEG_data']
    #df=pd.DataFrame(key_data)
    #print(df.describe)
    channels=key_data.shape[1]
    print(channels)
    print(f'plotting data from file:{file_name}')

sfreq=250.0
num_samples=25
time_indices=np.linspace(0,key_data.shape[0]-1,num_samples,endpoint=False,dtype=int)
sampled_data=np.zeros((num_samples,channels))

for file_name in file_names:
    data=scipy.io.loadmat(file_name)
    key_data=data['EEG_data']
    #df=pd.DataFrame(key_data)
    #print(df.describe)
    channels=key_data.shape[1]
    for ch in range(channels):
        sampled_data[:,ch]=key_data[time_indices,ch]
        corr_matrix=np.corrcoef(sampled_data,rowvar=False)
        #print(f'correlation matrix for file{file_name}:{corr_matrix}')

G=nx.Graph()
threshold=0.9

for i in range(channels):
    G.add_node(i)

for i in range(channels):
    for j in range(i+1,channels):
        if np.abs(corr_matrix[i,j])>threshold:
            G.add_edge(i,j,weight=corr_matrix[i,j])

if G.number_of_nodes()==0:
    print("graph is null")
else:
    pos=nx.spring_layout(G)
    edges = G.edges(data=True)
