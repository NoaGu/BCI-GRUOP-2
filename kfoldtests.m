clear
recordingFolder='C:\Recordings\Sub20'
FeaturesTrain = cell2mat(struct2cell(load(strcat(recordingFolder,'/FeaturesTrainSelected.mat'))))   % features for train set
LabelTrain = cell2mat(struct2cell(load(strcat(recordingFolder,'/LabelTrain'))));                % label vector for train set
length(LabelTrain)
size(FeaturesTrain)
% label vector
LabelTest = cell2mat(struct2cell(load(strcat(recordingFolder,'/LabelTest'))));      % label vector for test set
FeaturesTest=cell2mat(struct2cell(load(strcat(recordingFolder,'/FeaturesTest.mat')))); 
Labels = [LabelTrain,LabelTest]
Features = [transpose(FeaturesTrain), transpose(FeaturesTest)]
load ionosphere
tbl = array2table(transpose(Features));
tbl.Y = transpose(Labels);
n = length(Labels)
c = cvpartition(n, "holdout", 0.3)
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
%prediction = kfoldPredict(cvMdl1);
Mdl4 = fitcknn(tblTrain,'Y');
cvMdl4 = crossval(Mdl4); % Performs stratified 10-fold cross-validation
cvtrainError4 = kfoldLoss(cvMdl1)
%%
prediction = kfoldPredict(cvMdl3);

figure()
cm_ecoc = confusionchart(tblTrain.Y, prediction)
title(['KNN: train acc=' num2str(1-cvtrainError1)])

testError = loss(Mdl1,tblNew,'Y');
testAccuracy = 1-testError

tblTest = tblNew.Variables;
tblTest(:,11) = [];
testprediction = predict(Mdl1, tblTest);

figure()
cmt_ecoc = confusionchart(tblNew.Y, testprediction)
title(['SVM: test acc=' num2str(testAccuracy)])
%%
save('cvMdl3','cvMdl3')