% used for datasets with non-linear decison boundary, also datasets with
% full, not sparse, data matrix
clc
clear

global trainData testData testLabel
dname = 'scene';
load(['datasets/' dname '-train.mat']);
load(['datasets/' dname '-test.mat']);

epoch = 1;
times = 20;  % run 20 times for calculating mean accuracy
testTime = zeros(times,1);

[n,d] = size(trainData);
L = size(trainLabel,2);

scale = 2.^(1.25);
C = 2^(-3);

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


kernelMatrix = importdata([dname '/kernelMatrix_scale_' num2str(scale) '.mat']);

tStart = tic;
for run = 1:times
    coff = zeros(n, L+1);
    SVsIdx = zeros(1, n);
    SVsNum = 0;
    for o = 1:epoch
        index = randperm(n);
        for i=1:n
            j = index(i);
            x = trainData(j,:)';
            y = trainLabel(j,:);
            t = (o - 1)*n + i;
            
            if t == 1
                pred_v = zeros(1,L+1);
                km = [];
            else
                km = kernelMatrix(SVsIdx(1:SVsNum),j);
                pred_v = km'* coff(1:SVsNum,:);
            end
            
            % pred_y = pred_v(1:L) > pred_v(L+1); %online prediction
            maxIterNum = sum(y) * (L - sum(y));
            if maxIterNum == 0
                maxIterNum = L;
            end
            
            for iter = 1:maxIterNum
                snx = 1;
                kappa = 1.0 / (2 + 1.0 / (2*C*snx));
                r_t = -1;
                s_t = -1;
                minv = inf;
                maxv = -inf;
                for k = 1:L
                    if y(k) == 1
                        if pred_v(k) < minv
                            minv = pred_v(k);
                            r_t = k;
                        end
                    else
                        if pred_v(k) > maxv
                            maxv = pred_v(k);
                            s_t = k;
                        end
                    end
                end
                
                if r_t~= -1
                    f_t_1 = 1 - (pred_v(r_t) - pred_v(L+1));
                    l_t_1 = max(0, f_t_1);
                else
                    f_t_1 = NaN;
                    l_t_1 = 0;
                    r_t = 1;
                end
                
                if s_t ~= -1
                    f_t_2 = 1 - (pred_v(L+1) - pred_v(s_t));
                    l_t_2 = max(0, f_t_2);
                else
                    f_t_2 = NaN;
                    l_t_2 = 0;
                    s_t = 1;
                end
                
                if isnan(f_t_1)
                    alpha = 0;
                    beta = max(kappa * f_t_2 / snx, 0);
                elseif isnan(f_t_2)
                    alpha = max(kappa * f_t_1 / snx, 0);
                    beta = 0;
                elseif f_t_1 <=0 && f_t_2 <=0
                    alpha = 0;
                    beta = 0;
                elseif f_t_1 > 0 && f_t_2 <= -kappa * f_t_1
                    alpha = kappa * f_t_1 / snx;
                    beta = 0;
                elseif f_t_2 > 0 && f_t_1 <= -kappa * f_t_2
                    alpha = 0;
                    beta = kappa * f_t_2 / snx;
                else
                    alpha = (f_t_1 + kappa * f_t_2) / ((1.0/kappa - kappa) * snx);
                    beta = (kappa * f_t_1 + f_t_2) / ((1.0/kappa - kappa) * snx);
                end
                
                if alpha == 0 && beta == 0
                    break;
                end
                
                if o == 1
                    if iter == 1
                        SVsNum = SVsNum + 1;
                        SVsIdx(SVsNum) = j;
                        curId = SVsNum;
                        km = [km; 1];
                    end
                else
                    if iter == 1
                        id = find(SVsIdx(1:SVsNum) == j,1);
                        if ~isempty(id)
                            curId = id;
                        else
                            SVsNum = SVsNum + 1;
                            SVsIdx(SVsNum) = j;
                            curId = SVsNum;
                            km = [km; 1];
                        end
                    end
                end
                coff(curId, r_t) =  coff(curId, r_t) + alpha;
                coff(curId, s_t) =  coff(curId, s_t) - beta;
                coff(curId, L+1) =  coff(curId, L+1) - (alpha - beta);
                
                % re-compute the predicted value for label r_t, s_t, L+1
                pred_v(r_t) = pred_v(r_t) + km(curId) * alpha;
                pred_v(s_t) = pred_v(s_t) - km(curId) * beta;
                pred_v(L+1) = pred_v(L+1) - km(curId) * (alpha - beta);
            end
        end
    end
    
    tic
    %-------------evaluate model performance on test data-------------------------
    [test_macro_F1_score(run), test_micro_F1_score(run), hammingLoss(run), subsetAccuracy(run),  ...
        precision(run), recall(run), F1score(run), rankingLoss(run), oneError(run)] = testEvaluate_kernel_efficient(SVsIdx(1:SVsNum),coff,SVsNum,scale);
    clear coff SVsIdx
    testTime(run) = toc;
end

totalTime = toc(tStart);
avgTestTime = mean(testTime);
avgTrainTime = (totalTime - sum(testTime))/times;

clear kernelMatrix
%-------------output result to file----------------------------------------
fid = fopen('kernel_PAML_II_(running_time).txt','a');
fprintf(fid,'name = scene, kernel_PAML_II, runTimes = %d, epoch = %d, scale = %g, C = %g\n', times, epoch, scale, C);
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




