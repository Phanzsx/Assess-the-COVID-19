close all;
clear;

traindata = importdata('traindata.mat');
testdata = importdata('testdata.mat');
trainlabel = importdata('trainlabel.mat');
testlabel = importdata('testlabel.mat');
feats_importance = importdata('feat_import.mat');%the weights of 632 features
[sort_feats,index] = sort(feats_importance);


j = 17;%feature quantity
new_traindata = [];
new_testdata =[];
for i = 1:j
     one_train_feats = traindata(:,index(633-i));
     one_test_feats = testdata(:,index(633-i));
     new_traindata = [new_traindata,one_train_feats];
     new_testdata = [new_testdata,one_test_feats];
end
       
        %classification
        %Random Forest classifier
%         B = TreeBagger(288,new_traindata,trainlabel,'OOBPrediction','on','NumPredictorsToSample',2,'MinLeafSize',1);
%         [predict_label,pro,~] = predict(B,new_testdata);
%         for k = 1:length(testlabel)
%             pre(k,1) = str2num(predict_label{k,1});
%         end
        
        %KNN classifier
%         mdl = ClassificationKNN.fit(new_traindata,trainlabel,'NumNeighbors',1);
%         [pre,pro,~] = predict(mdl, new_traindata);
        
        %Naive Bayes
%                 nb = NaiveBayes.fit(new_traindata, trainlabel);
%                 [pre(j)] = predict(nb, new_testdata);
        
        %Ensembels classifier
%                 ens = fitensemble(new_traindata,trainlabel,'AdaBoostM2',100,'tree','type','classification');
%                 [pre,pro] = predict(ens, new_testdata);
%                 score(j) = pro(1);

       %discriminant analysis classifier
%                Factor = ClassificationDiscriminant.fit(new_traindata, trainlabel, 'discrimType', 'pseudoLinear');
%                [pre, Scores] = predict(Factor, new_testdata);
        
        %SVM classifier
        SVMStruct = svmtrain(trainlabel,new_traindata,'-s 0 -t 1 -c 1.0 -g 1.0 -b 1 ');
        [pre, accuracy, prob_estimates] = svmpredict(testlabel,new_testdata,SVMStruct,'-b 1');
        [label,rOrder] = sort(SVMStruct.Label);
        prob_estimatesR = prob_estimates(:,rOrder);
       
acc = sum((pre-testlabel) == 0)/length(testlabel);
disp(['Accuracy: ',num2str(acc)]);




