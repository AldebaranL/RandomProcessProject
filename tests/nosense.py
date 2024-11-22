import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os

current_directory=os.getcwd()
print(f"current director:{current_directory}")


file_names=[f'EEGsigsimagined_subjectP1_session20170901_block{i}.mat'for i in range(1,7)]


for file_name in file_names:
    try:
        data=scipy.io.loadmat(file_name)
        print(data.keys())
        eeg_data=file_names['EEG_data']
        print(f"Type and shape of eeg_data_array: {type(eeg_data_array)}, {eeg_data_array.shape}")

        if eeg_data is None:
            raise KeyError("EEG_data key not found in the file.")

        if not isinstance(eeg_data_array, np.ndarray):
              raise TypeError(f"eeg_data_array is not a numpy array, but {type(eeg_data_array)}")

        if len(eeg_data_array.shape) != 2:
            raise ValueError(f"eeg_data_array does not have 2 dimensions, but {eeg_data_array.shape}")

        plt.figure(figsize=(10,8))
        for i in range(eeg_data.shape[0]):
            plt.plot(eeg_data[i])

        plt.xlabel('time')
        plt.ylabel('eeg ')
        plt.show()
    except FileNotFoundError:
        print(f"File not found: {file_name}")
    except KeyError as ke:
        print(f"Key error: {ke}")
    except TypeError as te:
        print(f"Type error: {te}")
    except ValueError as ve:
        print(f"Value error: {ve}")
    except Exception as e:
        print(f"An error occurred while processing {file_name}: {e}")
