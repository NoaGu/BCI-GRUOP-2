%create traningVec from EE, run after MI2
recordingFolder='C:\Recordings\online2'
true_ind=find(strcmp({EEG.event.type},'3.000000000000000'))
false_ind=find(strcmp({EEG.event.type},'4.000000000000000'))
all_ind=[true_ind,false_ind]
all_ind=sort(all_ind)
events={EEG.event.type}
events=events(all_ind)
trainingVec=cellfun(@str2num,events)
trainingVec=trainingVec-2
fileName=[recordingFolder,'\trainingVec']
save(fileName ,'trainingVec')
