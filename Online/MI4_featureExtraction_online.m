function EEG_Features = MI4_featureExtraction(MIData,orgFolder,Fs)
%% This function extracts features for the machine learning process.
% Starts by visualizing the data (power spectrum) to find the best powerbands.
% Next section computes the best common spatial patterns from all available
% labeled training trials. The next part extracts all learned features.
% This includes a non-exhaustive list of possible features (commented below).
% At the bottom there is a simple feature importance test that chooses the
% best features and saves them for model training.


%% This code is part of the BCI-4-ALS Course written by Asaf Harel
% (harelasa@post.bgu.ac.il) in 2021. You are free to use, change, adapt and
% so on - but please cite properly if published.

%% Load previous variables:
load([orgFolder,'\SelectedIdx.mat']);                           % load the openBCI channel location
load([orgFolder,'\EEG_chans'])
disp(size(MIData))
MIData(1,:,:)=MIData
disp(size(MIData))
Fs = 125;                                                       % openBCI Cyton+Daisy by Bluetooth sample rate
[R, C] = size(EEG_chans);                                       % get EEG_chans (char matrix) size - rows and columns
chanLocs = reshape(EEG_chans',[1, R*C]);                        % reshape into a vector in the correct order
numChans = size(MIData,2);                                      % get number of channels from main data variable
numClasses=3
% Visual Feature Selection: Power Spectrum
% init cells for  Power Spectrum display
motorDataChan = {};
welch = {};
idxTarget = {};
freq.low = 0.5;                             % INSERT the lowest freq 
freq.high = 60;                             % INSERT the highst freq 
freq.Jump = 1;                              % SET the freq resolution
f = freq.low:freq.Jump:freq.high;           % frequency vector
window = 40;                                % INSERT sample size window for pwelch
noverlap = 20;                              % INSERT number of sample overlaps for pwelch
vizChans = [1,2];                           % INSERT which 2 channels you want to compare


psd = nan(numChans,numClasses,2,1000); % init psd matrix
for chan = 1:numChans
    motorDataChan{chan} = squeeze(MIData(:,chan,:))';                   % convert the data to a 2D matrix fillers by channel
    nfft = 2^nextpow2(size(motorDataChan{chan},1));                     % take the next power of 2 length of the specific trial length
    welch{chan} = pwelch(motorDataChan{chan},window, noverlap, f, Fs);  % calculate the pwelch for each electrode
    
    for class = 1:numClasses
        idxTarget{class} = find(targetLabels == class);                 % find the target index
                                            % add name of electrode
        for trial = 1:length(idxTarget{class})                          % run over all concurrent class trials
            [s,spectFreq,t,psd] = spectrogram(motorDataChan{chan}(:,idxTarget{class}(trial)),window,noverlap,nfft,Fs);  % compute spectrogram on specific channel
            multiPSD(trial,:,:) = psd;
        end
        
        % compute mean spectrogram over all trials with same target
        totalSpect(chan,class,:,:) = squeeze(mean(multiPSD,1));
        clear multiPSD psd
    end
end
%% Spectral frequencies and times for bandpower features:
% frequency bands
% bands{1} = [15.5,18.5];
% bands{2} = [8,10.5];
% bands{3} = [10,15.5];
% bands{4} = [17.5,20.5];
% bands{5} = [12.5,30];
% %times of frequency band features
% times{1} = (1*Fs : 2.5*Fs);
% times{2} = (1*Fs : 2.5*Fs);
% times{3} = (2.25*Fs : size(MIData,3));
% times{4} = (2*Fs : 2.5*Fs);
% times{5} = (1.5*Fs : 2.5*Fs);
%sub17
bands{1} = [15 ,20];
bands{2} = [20 ,25];
bands{3} = [25 ,30];
bands{4} = [12,15];
bands{5} = [25,30];
times{1}=(1 :1.5*Fs)
times{2}=(1 :1.5*Fs)
times{3}=(1.5*Fs :3*Fs)
times{4}=(1.5*Fs :3*Fs)
times{5}=(1.5*Fs :3*Fs)
%sub14
%bands{1} = [12 ,15];
%bands{2} = [20,25];
%bands{3} = [20,25];
%bands{4} = [27,32];
%bands{5} = [15,20];
%times{1}=(3.5*Fs :4.5*Fs)
%times{2}=(1 :1*Fs)
%times{3}=(1.*Fs :2*Fs)
%times{4}=(1*Fs :2*Fs)
%times{5}=(1*Fs :2*Fs)

numSpectralFeatures = length(bands);                        % how many features exist overall 

%% Extract features 
MIFeaturesLabel = NaN(1,numChans,numSpectralFeatures); % init features + labels matrix
n=0  
for channel = 1:numChans                        % run over all the electrodes (channels)
        n = 1;                                      % start a new feature index
        for feature = 1:numSpectralFeatures                 % run over all spectral band power features from the section above
            % Extract features: bandpower +-1 Hz around each target frequency
            MIFeaturesLabel(trial,channel,n) = bandpower(squeeze(MIData(trial,channel,times{feature}(1):times{feature}(end))),Fs,bands{feature});
            n = n+1;            
        end
        disp(strcat('Extracted Powerbands from electrode:',EEG_chans(channel,:)))
        
        
        % NOVEL Features - an explanation for each can be found in the class presentation folder
        
        % Normalize the Pwelch matrix
        pfTot = sum(welch{channel}(:,trial));               % Total power for each trial
        normlizedMatrix = welch{channel}(:,trial)./pfTot;   % Normalize the Pwelch matrix by dividing the matrix in its sum for each trial
        disp(strcat('Extracted Normalized Pwelch Matrix from electrode:',EEG_chans(channel,:)))
        
        
        % Root Total Power
        MIFeaturesLabel(trial,channel,n) = sqrt(pfTot);     % Square-root of the total power
        n = n + 1;
        disp(strcat('Extracted Root Total Power from electrode:',EEG_chans(channel,:)))
        
        
        % Spectral Moment
        MIFeaturesLabel(trial,channel,n) = sum(normlizedMatrix.*f'); % Calculate the spectral moment
        n = n + 1;
        disp(strcat('Extracted Normalized Pwelch Matrix from electrode:',EEG_chans(channel,:)))
        
        
        % Spectral Edge
        probfunc = cumsum(normlizedMatrix);                 % Create matrix of cumulative sum
        % frequency that 90% of the power resides below it and 10% of the power resides above it
        valuesBelow = @(z)find(probfunc(:,z)<=0.9);         % Create local function
        % apply function to each element of normlizedMatrix
        fun4Values = arrayfun(valuesBelow, 1:size(normlizedMatrix',1), 'un',0);
        lengthfunc = @(y)length(fun4Values{y})+1;           % Create local function for length
        % apply function to each element of normlizedMatrix
        fun4length = cell2mat(arrayfun(lengthfunc, 1:size(normlizedMatrix',1), 'un',0));
        MIFeaturesLabel(trial,channel,n) = f(fun4length);   % Insert it to the featurs matrix
        n = n + 1;
        disp(strcat('Extracted Spectral Edge from electrode:',EEG_chans(channel,:)))
        
        
        % Spectral Entropy
        MIFeaturesLabel(trial,channel,n) = -sum(normlizedMatrix.*log2(normlizedMatrix)); % calculate the spectral entropy
        n = n + 1;
        disp(strcat('Extracted Spectral Entropy from electrode:',EEG_chans(channel,:)))
        
        
        % Slope
        transposeMat = (welch{channel}(:,trial)');          % transpose matrix
        % create local function for computing the polyfit on the transposed matrix and the frequency vector
        FitFH = @(k)polyfit(log(f(1,:)),log(transposeMat(k,:)),1);
        % convert the cell that gets from the local func into matrix, perform the
        % function on transposeMat, the slope is in each odd value in the matrix
        % Apply function to each element of tansposeMat
        pFitLiner = cell2mat(arrayfun(FitFH, 1:size(transposeMat,1), 'un',0));
        MIFeaturesLabel(trial,channel,n)=pFitLiner(1:2 :length(pFitLiner));
        n = n + 1;
        disp(strcat('Extracted Slope from electrode:',EEG_chans(channel,:)))
        
        
        % Intercept
        % the slope is in each double value in the matrix
        MIFeaturesLabel(trial,channel,n)=pFitLiner(2:2:length(pFitLiner));
        n= n + 1;
        disp(strcat('Extracted Intercept from electrode:',EEG_chans(channel,:)))
        
        
        % Mean Frequency
        % returns the mean frequency of a power spectral density (PSD) estimate, pxx.
        % The frequencies, f, correspond to the estimates in pxx.
        MIFeaturesLabel(trial,channel,n) = meanfreq(normlizedMatrix,f);
        n = n + 1;
        disp(strcat('Extracted Mean Frequency from electrode:',EEG_chans(channel,:)))
        
        
        % Occupied bandwidth
        % returns the 99% occupied bandwidth of the power spectral density (PSD) estimate, pxx.
        % The frequencies, f, correspond to the estimates in pxx.
        MIFeaturesLabel(trial,channel,n) = obw(normlizedMatrix,f);
        n = n + 1;
        disp(strcat('Extracted Occupied bandwidth from electrode:',EEG_chans(channel,:)))
        
        
        % Power bandwidth
        MIFeaturesLabel(trial,channel,n) = powerbw(normlizedMatrix,Fs);
        n = n + 1;
        disp(strcat('Extracted Power bandwidth from electrode:',EEG_chans(channel,:)))
        
    end


% z-score all the features
MIFeaturesLabel = zscore(MIFeaturesLabel);

% Reshape into 2-D matrix
MIFeatures = reshape(MIFeaturesLabel,1,[]);
EEG_Features=MIFeatures(SelectedIdx-3)
end
 

