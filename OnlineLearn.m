function trainedModel = OnlineLearn(recordingFolder,orgFolder,lengthOfTrial,trial_num,cueVec);
%trainedModel=trainedModel
Fs=125
trainingVec=cueVec(1:trial_num)

onsetVec=[1:lengthOfTrial*Fs:lengthOfTrial*Fs*length(trainingVec)];
for i=1:length(onsetVec)
    EEG_event(i).type=1111;
    EEG_event(i).latency=onsetVec(i);
    EEG_event(i).duration=1;
end
    save(strcat(recordingFolder,'\','EEG_events.mat'),'EEG_event');
save(strcat(recordingFolder,'\trainingVec.mat'),'trainingVec');
MI3_segmentation(recordingFolder)
load([recordingFolder,'\MIData.mat']);
MIData_new=MIData;
disp('MIData new size:')
size(MIData_new)

load([orgFolder,'\MIData.mat']);
MIData_old=MIDataAll;
disp('MIData old size:')
size(MIData_old)
MIData=[MIData_old;MIData_new];
disp('MIData all size:')
size(MIData)

trainingVec=cell2mat(struct2cell(load(strcat(orgFolder,'\trainingVec'))));
trainingVec=[trainingVec , cueVec(1:trial_num)];
save(strcat(recordingFolder,'\trainingVec.mat'),'trainingVec');
save(strcat(recordingFolder,'\MIData.mat'),'MIData');
MI4_featureExtraction(recordingFolder);
trainedModel=kfoldtests(recordingFolder);
end
