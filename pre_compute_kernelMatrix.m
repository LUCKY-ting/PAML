clc
clear
load('datasets/scene-train.mat');

[n,d] = size(trainData);
L = size(trainLabel,2);

para_matrix = 2.^(1:0.25:2);

tic
for p = 1: size(para_matrix, 2)  %pre-compute the kernel matrix: km(i,j) = \phi(xi)*\phi(xj)
    scale = para_matrix(p);
    [kernelMatrix,~] = rbfkernel_call(trainData, scale);
    save(['scene/kernelMatrix_scale_' num2str(scale) '.mat'],'kernelMatrix');
    disp(['success!  scale =' num2str(scale)]);
    clear kernelMatrix
end
toc
