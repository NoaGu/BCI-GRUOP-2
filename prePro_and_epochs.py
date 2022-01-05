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
import numpy as np
import matplotlib.pyplot as plt

from mne import Epochs, create_info, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.time_frequency import AverageTFR
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from mne.decoding import SlidingEstimator, cross_val_multiscore, get_coef,  LinearModel
#%%
import numpy as np
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
def set_reference_digitization(raw):
    raw.pick_types(meg=False, eeg=True, eog=True, ecg=True, stim=True,
                   exclude=raw.info['bads']).load_data()
    raw.set_eeg_reference(projection=True).apply_proj()
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(ten_twenty_montage)
    return raw

fname = 'C:\Recordings\Sub17\EEG.xdf'
streams, header = pyxdf.load_xdf(fname)
#streams[0]=streams1[1]
#streams[1]=streams1[0]
events=np.zeros([len(streams[1]["time_series"]),3])
events[:,0]=np.transpose(streams[1]["time_stamps"])
for i in np.arange(len(streams[1]["time_stamps"])):
    c=np.where(find_nearest(streams[0]["time_stamps"],streams[1]["time_stamps"][i])==streams[0]["time_stamps"])
    events[i,0]=c[0]
events[:,2]=np.transpose(streams[1]["time_series"])

event_id={'left':1,'right':2,'idle':3}
#events=mne.pick_events([1.,2.,3.])
data = np.transpose(streams[0]["time_series"][:,0:13])
chs=['C3','C4','Cz','FC1',
    'FC2','FC5','FC6','CP1', 'CP2',
    'CP5','CP6','O1','O2']
sfreq = float(streams[0]["info"]["nominal_srate"][0])
data*= (1e-6 / 50 / 2)
info = mne.create_info(chs, sfreq, 'eeg')
raw = mne.io.RawArray(data,info)
raw=set_reference_digitization(raw)
#raw.set_montage()
#anno=mne.annotations_from_events(events, sfreq)
#raw.set_annotations(anno)
#%%
raw.plot()
        #%%
raw=raw.filter(1.,40.)

#%%

#%%
events=events.astype(int)
epochs=mne.Epochs(raw,events,event_id=event_id,tmin=-1.,tmax=2.5,baseline=None)
epochs.plot()
#%%
ica = ICA(n_components=12, max_iter='auto', random_state=97)
ica.fit(epochs)
#ica
#ica.plot_components()
#ica.plot_properties(epochs,np.arange(12))
#%%
ica.exclude = [] 

#ica.apply(epochs)

#%%
evoked_left=epochs['left'].average()
evoked_left.plot(titles='left',spatial_colors=True)
evoked_right=epochs['right'].average()
evoked_right.plot(titles='right',spatial_colors=True)
evoked_right=epochs['idle'].average()
evoked_right.plot(titles='idle',spatial_colors=True)
#%%
#evoked_right.plot_topomap([0,0.1,0.5,1.,1.5,2.,3.,4.])
#evoked_left.plot_topomap([0,0.1,0.5,1.,1.5,2.,3.,4.])
#%%
tfr_freqs = np.arange(5., 40., 3.)
window_size = 0.3
#tfr_freqs = np.logspace(0.7, 2.2, 25)
tfr_cycles = tfr_freqs * window_size
freq_baseline = (-0.5,0.)
condition=['left','right','idle']
for i in np.arange(len(condition)):
 this_epochs = epochs[condition[i]]
 this_epochs.apply_baseline((-0.5,0))
 power, itc = mne.time_frequency.tfr_morlet(
                this_epochs, freqs=tfr_freqs, return_itc=True, n_cycles=tfr_cycles)
 tmp = power.copy().apply_baseline(baseline=freq_baseline, mode='logratio').data
 clim = np.quantile(abs(tmp), 0.95)  # another option is just to crop the edges, will probably fix some of the issue
 fig = power.plot_topo(baseline=freq_baseline, mode='logratio', title='TFR '+condition[i])#), vmin=-clim, vmax=clim)
#%%
clf = make_pipeline(StandardScaler(),
        LinearModel(LogisticRegression(solver='liblinear')))
n_splits = 5  # for cross-validation, 5 is better, here we use 3 for speed
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Classification & time-frequency parameters
tmin, tmax = 0.1, 2.5
n_cycles = 3.  # how many complete cycles: used to define window size
min_freq = 10.
max_freq = 30.
n_freqs = 6  # how many frequency bins to use

# Assemble list of frequency range tuples
freqs = np.linspace(min_freq, max_freq, n_freqs)  # assemble frequencies
freq_ranges = list(zip(freqs[:-1], freqs[1:]))  # make freqs list of tuples

# Infer window spacing from the max freq and number of cycles to avoid gaps
#%%
window_spacing = 1
centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]
n_windows = len(centered_w_times)
#%%
# Instantiate label encoder
le = LabelEncoder()
freq_scores = np.zeros((n_freqs - 1,))
event_id={'left':1,'right':2}

plt.figure()
# Loop through each frequency range of interest
for freq, (fmin, fmax) in enumerate(freq_ranges):

    # Infer window size based on the frequency being used
    w_size = n_cycles / ((fmax + fmin) / 2.)  # in seconds

    # Apply band-pass filter to isolate the specified frequencies
    raw_filter = raw.copy().filter(fmin, fmax, n_jobs=1, fir_design='firwin',
                                   skip_by_annotation='edge')

    # Extract epochs from filtered data, padded by window size
    epochs = Epochs(raw_filter, events, event_id, tmin - w_size, tmax + w_size,
                    proj=False, baseline=None, preload=True)
    epochs.drop_bad()
    y = le.fit_transform(epochs.events[:, 2])

    X = epochs.get_data()
    X=np.mean(X,2)
    # Save mean scores over folds for each frequency and time window
    freq_scores[freq] = np.mean(cross_val_score(estimator=clf, X=X, y=y,
                                                scoring='roc_auc', cv=cv,
                                                n_jobs=1), axis=0)
    #%%
    
    plt.bar(freqs[:-1], freq_scores, width=np.diff(freqs)[0],
        align='edge', edgecolor='black')
plt.xticks(freqs)
plt.ylim([0, 1])
plt.axhline(len(epochs['left']) / len(epochs), color='k', linestyle='--',
            label='chance level')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Decoding Scores')
plt.title('Frequency Decoding Scores')
#%%
# init scores
tf_scores = np.zeros((n_freqs - 1, 4))

# Loop through each frequency range of interest
for freq, (fmin, fmax) in enumerate(freq_ranges):

    # Infer window size based on the frequency being used
    w_size = 1.  # in seconds

    # Apply band-pass filter to isolate the specified frequencies
    raw_filter = raw.copy().filter(fmin, fmax, n_jobs=1, fir_design='firwin',
                                   skip_by_annotation='edge')

    # Extract epochs from filtered data, padded by window size
    epochs = Epochs(raw_filter, events, event_id, tmin - w_size, tmax + w_size,
                    proj=False, baseline=None, preload=True)
    epochs.drop_bad()
    y = le.fit_transform(epochs.events[:, 2])

    # Roll covariance, csp and lda over time
    for t, w_time in enumerate(centered_w_times):

        # Center the min and max of the window
        w_tmin = w_time - w_size / 2.
        w_tmax = w_time + w_size / 2.

        # Crop data into time-window of interest
        X = epochs.copy().crop(w_tmin, w_tmax).get_data()
        X=np.mean(X,2)
        # Save mean scores over folds for each frequency and time window
        tf_scores[freq, t] = np.mean(cross_val_score(estimator=clf, X=X, y=y,
                                                     scoring='roc_auc', cv=cv,
                                                     n_jobs=1), axis=0)
        #%%
        # Set up time frequency object
av_tfr = AverageTFR(create_info(['freq'], sfreq), tf_scores[np.newaxis, :],
                    centered_w_times, freqs[1:], 1)

chance = np.mean(y)  # set chance level to white in the plot
av_tfr.plot([0], vmin=chance, title="Time-Frequency Decoding Scores",
            cmap=plt.cm.Reds)
#%%
tf_scores=np.mean(tf_scores,1)