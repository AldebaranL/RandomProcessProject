import pandas as pd
import scipy.io
import os
import numpy
import matplotlib.pyplot as plt

current_directory=os.getcwd()
print(f"current director:{current_directory}")
file_names=[f'EEGsigsimagined_subjectP1_session20170901_block{i}.mat'for i in range(1,7)]

for file_name in file_names:
    data=scipy.io.loadmat(file_name)
    #print(data.keys())
    key_data=data['EEG_data']
    #print(key_data)
    channel=data['channel_names']
    #print(key_data.shape)
    #print(channel)
    timeofstudy=data['timeofstudy']
    #print(timeofstudy)
    prompt_time=data['prompt_start_time_marker']
    print(prompt_time)