'''
画图展示原始eeg数据
'''

import pandas as pd
import scipy.io
import os
import numpy
import matplotlib.pyplot as plt

current_directory=os.getcwd()
print(f"current director:{current_directory}")
file_names=[f'data/EEGsigsimagined_subjectP1_session20170901_block{i}.mat'for i in range(1,6)]+[f'data/DATASET/EEGsigsactive_subjectP2_session20230927_block{i}.mat' for i in range(1, 6)]

for file_index,file_name in enumerate(file_names):
    data=scipy.io.loadmat(file_name)
    print(data.keys())
    key_data=data['EEG_data']
    key2_data=data['channel_names']
    print(key_data.shape)
    print(key2_data)

    data=scipy.io.loadmat(file_name)
    key_data=data['EEG_data']
    #df=pd.DataFrame(key_data)
    #print(df.describe)
    channels=key_data.shape[1]
    print(channels)
    print(f'plotting data from file:{file_name}')

    plt.figure(figsize=(20,15))

    for i in range(0,24):
        plt.subplot(6,4,i+1)
        plt.plot(key_data[:,i])
        plt.title(f'channel{i+1}')
        plt.xlabel('time')
        plt.ylabel('amplitude')
    
    plt.tight_layout()
    plt.savefig(str(file_index)+'.png')
    plt.show()
    
    