import scipy.io
import os
import numpy as np
import mne

import matplotlib.pyplot as plt

file_names=[f'EEGsigsimagined_subjectP1_session20170901_block{i}.mat'for i in range(1,7)]

data=scipy.io.loadmat(file_names[0])
print(data['channel_names'])
channel_names=data['channel_names'].flatten()

channel_names = [str(ch[0]) if isinstance(ch, np.ndarray) else str(ch) for ch in channel_names]
montage=mne.channels.make_standard_montage('standard_1020')

info=mne.create_info(ch_names=list(channel_names),sfreq=250)
fake_data=np.zeros((len(channel_names),1))
raw=mne.io.RawArray(data=fake_data,info=info)

raw.set_montage(montage)
dig_loc=montage.get_positions()['ch_pos']
try:
    xy = np.array([dig_loc[ch] if ch in dig_loc else np.array([0,0])for ch in channel_names])[:, :2]
except KeyError as e:
    print(f'missing key{e} in dig_loc')
print("xy:",xy)
plt.figure(figsize=(10, 10))
plt.scatter(xy[:, 0], xy[:, 1], s=100, color='red')

for i, ch in enumerate(channel_names):
    plt.text(xy[i, 0], xy[i, 1], ch, fontsize=12, ha='right')
plt.title('pic')
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.grid(True)
plt.show()