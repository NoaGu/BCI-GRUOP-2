# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 14:55:43 2021

@author: User
"""

import os.path as op

import pyxdf
import numpy as np
import mne
from mne.datasets import misc
#%%
import numpy as np
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

fname = 'C:\Recordings\Sub12\EEG.xdf'
streams, header = pyxdf.load_xdf(fname)
events=np.zeros([len(streams[0]["time_series"]),3])
events[:,0]=np.transpose(streams[0]["time_stamps"])
for i in np.arange(len(streams[0]["time_stamps"])):
    c=np.where(find_nearest(streams[1]["time_stamps"],streams[0]["time_stamps"][i])==streams[1]["time_stamps"])
    events[i,0]=c[0]
events[:,2]=np.transpose(streams[0]["time_series"])
#%%
event_id={'left':1,'right':2,'idle':3}
#events=mne.pick_events([1.,2.,3.])
data = np.transpose(streams[1]["time_series"][:,0:13])
chs=['C03','C04','C0Z','FC1',
    'FC2','FC5','F06','CP1', 'CP2',
    'CP5','CP6','O01','O02']
sfreq = float(streams[1]["info"]["nominal_srate"][0])
data*= (1e-6 / 50 / 2)
info = mne.create_info(chs, sfreq, 'eeg')
raw = mne.io.RawArray(data, info)
#raw.set_montage()
#anno=mne.annotations_from_events(events, sfreq)
#raw.set_annotations(anno)
#%%
raw.plot()
        #%%
raw=raw.filter(1.,40.)
events=events.astype(int)
epochs=mne.Epochs(raw,events,event_id=event_id,tmin=-1.,tmax=5.)
#%%
evoked_left=epochs['left'].average()
evoked_left.plot()
evoked_right=epochs['right'].average()
evoked_right.plot()
#%%
