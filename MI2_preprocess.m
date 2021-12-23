function EEG = MI2_preprocess(recordingFolder)
%% Offline Preprocessing
% Assumes recorded using Lab Recorder.
% Make sure you have EEGLAB installed with ERPLAB & loadXDF plugins.

% [recordingFolder] - where the EEG (data & meta-data) are stored.

% Preprocessing using EEGLAB function.
% 1. load XDF file (Lab Recorder LSL output)
% 2. look up channel names - YOU NEED TO UPDATE THIS
% 3. filter data above 0.5 & below 40 Hz
% 4. notch filter @ 50 Hz
% 5. advanced artifact removal (ICA/ASR/Cleanline...) - EEGLAB functionality

%% This code is part of the BCI-4-ALS Course written by Asaf Harel
% (harelasa@post.bgu.ac.il) in 2021. You are free to use, change, adapt and
% so on - but please cite properly if published.

%% Some parameters (this needs to change according to your system):
addpath 'C:\Users\User\Desktop\bci\eeglab2021.1'           % update to your own computer path
eeglab;                                     % open EEGLAB 
highLim = 40;                               % filter data under 40 Hz
lowLim = 0.5;                               % filter data above 0.5 Hz
recordingFile = strcat(recordingFolder,'\EEG.XDF');

% (1) Load subject data (assume XDF)
EEG = pop_loadxdf(recordingFile, 'streamtype', 'EEG', 'exclude_markerstreams', {});
EEG.setname = 'MI_sub';

% (2) Update channel names - each group should update this according to
% their own openBCI setup.
EEG_chans(1,:) = 'C03';
EEG_chans(2,:) = 'C04';
EEG_chans(3,:) = 'CZ0';
EEG_chans(4,:) = 'FC1';
EEG_chans(5,:) = 'FC2';
EEG_chans(6,:) = 'FC5';
EEG_chans(7,:) = 'FC6';
EEG_chans(8,:) = 'CP1';
EEG_chans(9,:) = 'CP2';
EEG_chans(10,:) = 'CP5';
EEG_chans(11,:) = 'CP6';
EEG_chans(12,:) = 'OO1';
EEG_chans(13,:) = 'OO2';
EEG_chans(14,:) = 'P03';
EEG_chans(15,:) = 'P03';
EEG_chans(16,:) = 'P03';
figure()
plot(EEG.data')
psd=abs(fft(EEG.data'));
title('raw')
figure()
xlabels=(1:length(EEG.times))*(125/length(EEG.times))';
plot(xlabels(1:(length(EEG.times))/2),log(psd(1:(length(EEG.times))/2)))
%% high pass
EEG = pop_eegfiltnew(EEG, 'locutoff',lowLim,'plotfreqz',1);     % remove data under
EEG = eeg_checkset( EEG );
figure()
psd=abs(fft(EEG.data'));
figure()
xlabels=(1:length(EEG.times))*(125/length(EEG.times))';
plot(xlabels(1:(length(EEG.times))/2),psd(1:(length(EEG.times))/2))
title('high pass')
%% (3) Low-pass filter
EEG = pop_eegfiltnew(EEG, 'hicutoff',highLim,'plotfreqz',1);    % remove data above
EEG = eeg_checkset( EEG );
psd=abs(fft(EEG.data'));
figure()
xlabels=(1:length(EEG.times))*(125/length(EEG.times))';
plot(xlabels(1:(length(EEG.times))/2),psd(1:(length(EEG.times))/2))
title('Low pass')
% (3) High-pass filter

% (4) Notch filter - this uses the ERPLAB filter
EEG  = pop_basicfilter( EEG,  1:15 , 'Boundary', 'boundary', 'Cutoff',  50, 'Design', 'notch', 'Filter', 'PMnotch', 'Order',  180 );
EEG = eeg_checkset( EEG );
psd=abs(fft(EEG.data'));
figure()
xlabels=(1:length(EEG.times))*(125/length(EEG.times))';
plot(xlabels(1:(length(EEG.times))/2),psd(1:(length(EEG.times))/2))
title('notch')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% (5) Add advanced artifact removal functions %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C3_lap=EEG.data(1,:)-(sum(EEG.data([4,6,10,8],:))/4);
figure
plot(C3_lap)
title('C3 lap')
C4_lap=EEG.data(2,:)-(sum(EEG.data([5,7,9,11],:))/4);
figure
plot(C4_lap)
title('C4 lap')
%EEG.data(1,:)=C3_lap
%EEG.data(2,:)-C4_lap
% Save the data into .mat variables on the computer
EEG_data = EEG.data;            % Pre-processed EEG data
EEG_event = EEG.event;          % Saved markers for sorting the data
save(strcat(recordingFolder,'\','cleaned_sub.mat'),'EEG_data');
save(strcat(recordingFolder,'\','EEG_events.mat'),'EEG_event');
save(strcat(recordingFolder,'\','EEG_chans.mat'),'EEG_chans');
                
end
