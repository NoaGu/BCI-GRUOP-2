clear
recordingFolder='C:/Recordings/Sub17'
FeaturesTrain = cell2mat(struct2cell(load(strcat(recordingFolder,'\FeaturesTrainSelected.mat'))))   % features for train set
LabelTrain = cell2mat(struct2cell(load(strcat(recordingFolder,'\LabelTrain'))));                % label vector for train set
length(LabelTrain)
size(FeaturesTrain)
% label vector
LabelTest = cell2mat(struct2cell(load(strcat(recordingFolder,'\LabelTest'))));      % label vector for test set
FeaturesTest=cell2mat(struct2cell(load(strcat(recordingFolder,'\FeaturesTest.mat')))); 



%FeaturesTest=zscore(FeaturesTest)
%% plot the featuers
figure()
for i=1:length(LabelTrain)
 colors={'r','b','g'}
 scatter3(FeaturesTrain(i,1),FeaturesTrain(i,2),FeaturesTrain(i,3),colors{LabelTrain(i)})
 hold on
end
title('FeaturesTrain - 3 first Features')
legend({'left','right','Idle'})
X=pca(FeaturesTrain);
FeaturesTrain_pca=FeaturesTrain*X;
figure()
for i=1:length(LabelTrain)
 colors={'r','b','g'}
 scatter3(FeaturesTrain_pca(i,1),FeaturesTrain_pca(i,2),FeaturesTrain_pca(i,3),colors{LabelTrain(i)})
 hold on
end
title('PCA')
%%
SVMModels = cell(3,1);
classes = unique(LabelTrain);
rng(1); % For reproducibility

for j = 1:numel(classes)
    indx = (LabelTrain==classes(j)); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(FeaturesTrain,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
     %SVMModels{j} = fitcsvm(FeaturesTrain,indx,'ClassNames',[false true],'OptimizeHyperparameters','auto', ...
    %'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName', ...
    %'expected-improvement-plus'))
end
%SVMModels is a 3-by-1 cell array, with each cell containing a ClassificationSVM classifier. For each cell, the positive class is setosa, versicolor, and virginica, respectively.

%Define a fine grid within the plot, and treat the coordinates as new observations from the distribution of the training data. Estimate the score of the new observations using each classifier.
%%
N=length(FeaturesTest)
Scores = zeros(N,numel(classes));

for j = 1:numel(classes)
    [~,score] = predict(SVMModels{j},FeaturesTest);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end

[~,maxScore] = max(Scores,[],2);

acc=0
for i=1:N
    if maxScore(i)==LabelTest(i)
        acc=acc+1
    end
end
acc=acc/N
figure()
cm_svm = confusionchart(LabelTest,maxScore);
title(['svm: acc=' num2str(acc)])
%%
Mdl = fitcnb(FeaturesTrain,LabelTrain)
label = predict(Mdl,FeaturesTest)
acc=0
for i=1:N
    if label(i)==LabelTest(i)
        acc=acc+1
    end
end
acc=acc/N
figure()
cm_nb = confusionchart(LabelTest,label);
title(['nb: acc=' num2str(acc)])
%%
tree = fitctree(FeaturesTrain,LabelTrain)
label = predict(tree,FeaturesTest)
acc=0
for i=1:N
    if label(i)==LabelTest(i)
        acc=acc+1
    end
end
acc=acc/N
figure()
cm_tree = confusionchart(LabelTest,label);
title(['tree: acc=' num2str(acc)])