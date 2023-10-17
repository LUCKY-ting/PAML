clc
clear
global testData testLabel L;

load('../datasets/bibtex-train.mat');
load('../datasets/bibtex-test.mat');
epoch = 1;
times = 20;  % run 20 times for calculating mean accuracy
testTime = zeros(times,1);

[n,d] = size(trainData);
L = size(trainLabel,2);

trainData = sparse(trainData);
trainLabel = sparse(trainLabel);
testData = sparse(testData);
testLabel = sparse(testLabel);

sr = RandStream.create('mt19937ar','Seed',1);
RandStream.setGlobalStream(sr);

test_macro_F1_score = zeros(times,1);
test_micro_F1_score = zeros(times,1);
hammingLoss = zeros(times,1);
rankingLoss = zeros(times,1);
subsetAccuracy = zeros(times,1);
oneError = zeros(times,1);
precision = zeros(times,1);
recall = zeros(times,1);
F1score = zeros(times,1);
    
tStart = tic;
for run = 1:times
    index = randperm(n);
    w = PAML_sparse(trainData',trainLabel', index, epoch);
    
    tic
    [test_macro_F1_score(run), test_micro_F1_score(run), hammingLoss(run), subsetAccuracy(run),  ...
        precision(run), recall(run), F1score(run), rankingLoss(run), oneError(run)] = testEvaluate_efficient(w);
    testTime(run) = toc;
end

totalTime = toc(tStart);
avgTestTime = mean(testTime);
avgTrainTime = (totalTime - sum(testTime))/times;

%-------------output result to file----------------------------------------
fid = fopen('PAML(running_time).txt','a');
fprintf(fid,'name = bibtex, PAML, runTimes = %d, epoch = %d \n', times, epoch);
fprintf(fid,'macro_F1score +std, micro_F1score +std \n');
fprintf(fid,'%.4f, %.4f, %.4f, %.4f \n ', mean(test_macro_F1_score), std(test_macro_F1_score), mean(test_micro_F1_score), std(test_micro_F1_score));
fprintf(fid,'hammingloss +std, subsetAccuracy +std \n');
fprintf(fid,'%.4f, %.4f, %.4f, %.4f,\n ', mean(hammingLoss), std(hammingLoss), mean(subsetAccuracy), std(subsetAccuracy));
fprintf(fid,'precision +std,  recall +std,  F1score +std \n');
fprintf(fid,'%.4f, %.4f, %.4f, %.4f, %.4f, %.4f,\n', mean(precision), std(precision), mean(recall), std(recall), mean(F1score), std(F1score));
fprintf(fid,'rankingLoss +std, oneErr +std \n');
fprintf(fid,'%.4f, %.4f, %.4f, %.4f \n ', mean(rankingLoss), std(rankingLoss), mean(oneError), std(oneError));
fprintf(fid,'training time [s], testing time [s]\n');
fprintf(fid,'%.4f, %.4f \n\n', avgTrainTime, avgTestTime);
fclose(fid);


