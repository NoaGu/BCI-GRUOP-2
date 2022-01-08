clear
recordingFolder='/Users/wrystrn/Documents/BCI4ALS/recordings/Sub17'
FeaturesTrain = cell2mat(struct2cell(load(strcat(recordingFolder,'/FeaturesTrainSelected.mat'))))   % features for train set
LabelTrain = cell2mat(struct2cell(load(strcat(recordingFolder,'/LabelTrain'))));                % label vector for train set
length(LabelTrain)
size(FeaturesTrain)
% label vector
LabelTest = cell2mat(struct2cell(load(strcat(recordingFolder,'/LabelTest'))));      % label vector for test set
FeaturesTest=cell2mat(struct2cell(load(strcat(recordingFolder,'/FeaturesTest.mat')))); 
Data = [LabelTrain,LabelTest]
Features = [transpose(FeaturesTrain), transpose(FeaturesTest)]
load ionosphere
tbl = array2table(transpose(Features));
tbl.Y = transpose(Data);
n = length(Data)
c = cvpartition(n, "Holdout", 10)
idxTrain = training(c);
tblTrain = tbl(idxTrain,:);
idxNew = test(c);
tblNew = tbl(idxNew,:);
Mdl2 = fitcnb(tblTrain,'Y');
cvMdl2 = crossval(Mdl2); % Performs stratified 10-fold cross-validation
cvtrainError2 = kfoldLoss(cvMdl2)
Mdl3 = fitctree(tblTrain,'Y');
cvMdl3 = crossval(Mdl3); % Performs stratified 10-fold cross-validation
cvtrainError3 = kfoldLoss(cvMdl3)
Mdl1 = fitcecoc(tblTrain,'Y');
cvMdl1 = crossval(Mdl1); % Performs stratified 10-fold cross-validation
cvtrainError1 = kfoldLoss(cvMdl1)
