%% connect runs
runs_name=[110,111,200,300]
x=[]
trainingVec=[]
new_folder='C:\Recordings\Sub400'
%mkdir(new_folder)
for i=1:length(runs_name)
    recordingFolder=['C:\Recordings\Sub',num2str(runs_name(i))]
    load(strcat(recordingFolder,'\MIData.mat'));
    targetLabels = cell2mat(struct2cell(load(strcat(recordingFolder,'\trainingVec'))));
    length(targetLabels)
    idx=find(targetLabels==3)
    targetLabels(idx)=[];
    MIData(idx,:,:)=[];
    x=[x;MIData(1:length(targetLabels),:,1:313)];
    
    trainingVec=[trainingVec,targetLabels];
end
MIData=x;
cd(new_folder)
save('MIData','MIData')
save('trainingVec','trainingVec')
