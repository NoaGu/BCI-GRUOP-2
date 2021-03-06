function Mdl2=kfoldtests(recordingFolder)
%you can define here costs and priors 
cost=[0,1;3,0] %cost i,j is the cost of clssify group i as j.
prior=[0.66,0.33] %prob of classes
%read the data
FeaturesTrain = cell2mat(struct2cell(load(strcat(recordingFolder,'/FeaturesTrainSelected.mat'))))   % features for train set
LabelTrain = cell2mat(struct2cell(load(strcat(recordingFolder,'/LabelTrain'))));                % label vector for train set
LabelTest = cell2mat(struct2cell(load(strcat(recordingFolder,'/LabelTest'))));      % label vector for test set
FeaturesTest=cell2mat(struct2cell(load(strcat(recordingFolder,'/FeaturesTest.mat')))); 
Labels = [LabelTrain,LabelTest]
Features = [transpose(FeaturesTrain), transpose(FeaturesTest)]

load ionosphere
%create test set and train set
tbl = array2table(transpose(Features));
tbl.Y = transpose(Labels);
n = length(Labels)
c = cvpartition(n, "holdout", 0.3)
idxTrain = training(c);
tblTrain = tbl(idxTrain,:);
idxNew = test(c);
tblNew = tbl(idxNew,:);
%naive bayes
Mdl2 = fitcnb(tblTrain,'Y',prior=prior);
cvMdl2 = crossval(Mdl2); % Performs stratified 10-fold cross-validation
cvtrainError2 = kfoldLoss(cvMdl2)
%tree
Mdl3 = fitctree(tblTrain,'Y');
cvMdl3 = crossval(Mdl3); % Performs stratified 10-fold cross-validation
cvtrainError3 = kfoldLoss(cvMdl3)
%svm
Mdl1 = fitcecoc(tblTrain,'Y',prior=prior);
cvMdl1 = crossval(Mdl1); % Performs stratified 10-fold cross-validation
cvtrainError1 = kfoldLoss(cvMdl1)
%knn
Mdl4 = fitcknn(tblTrain,'Y',cost=cost);
cvMdl4 = crossval(Mdl4); % Performs stratified 10-fold cross-validation
cvtrainError4 = kfoldLoss(cvMdl1)
%% plot the resultes (test set)


testError = loss(Mdl1,tblNew,'Y');
testAccuracy = 1-testError

tblTest = tblNew.Variables;
tblTest(:,11) = [];
testprediction = predict(Mdl1, tblTest);

figure()
cmt_ecoc = confusionchart(tblNew.Y, testprediction)
title(['SVM: test acc=' num2str(testAccuracy)])
%%
testError = loss(Mdl4,tblNew,'Y');
testAccuracy = 1-testError

tblTest = tblNew.Variables;
tblTest(:,11) = [];
testprediction = predict(Mdl4, tblTest);

figure()
cmt_ecoc = confusionchart(tblNew.Y, testprediction)
%cmt_ecoc.ColumnSummary = 'column-normalized';
cmt_ecoc.RowSummary = 'row-normalized';
title(['KNN'])%: test acc=' num2str(testAccuracy)])
%%
testError = loss(Mdl2,tblNew,'Y');
testAccuracy = 1-testError

tblTest = tblNew.Variables;
tblTest(:,11) = [];
testprediction = predict(Mdl2, tblTest);

figure()
cmt_ecoc = confusionchart(tblNew.Y, testprediction)
title(['NB: test acc=' num2str(testAccuracy)])
%% save the models
save('cvMdl3','cvMdl3')
save('Mdl3','Mdl4')