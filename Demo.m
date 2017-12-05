
%************************************************************************
% Non-parallel LSSVM Classifier Demo:

% Created by
%     Siamak Mehrkanoon
%     Dept. of Electrical Engineering (ESAT)
%     Research Group: STADIUS
%     KU LEUVEN
%
% (c) 2014
%************************************************************************

% Citation:

%[1] S. Mehrkanoon, X. Huang, J.A.K. Suykens,
% "Non-parallel Classifiers with Different Loss Functions",
% Neurocomputing, vol. 143, Nov. 2014, pp. 294-301.

%Author: Siamak Mehrkanoon

%%  ================= Example 1 of the Ref [1]  ===================


%-------------------------------------------------------------------------
clear all; close all; clc

%% load the dataset
load data
X1=data.X1; % data of class 1
X2=data.X2; % data of class 2
Y=data.Y; % labels
X=[X1 ; X2];
Xtest=data.Xtest; % test set


%% 5-fold CV
numFolds=5;
Indices = crossvalind('Kfold', size(X,1), numFolds);
for k=1:numFolds
    Ytr{k} = Y(Indices ~= k);
    Yval{k} = Y(Indices == k);
    Xtr{k} = X(Indices ~= k,:);
    Xval{k} = X(Indices == k,:);
    mask1=Ytr{k}>0;
    X1tr{k}=Xtr{k}(mask1,:);
    X2tr{k}=Xtr{k}(~mask1,:);
end


%% Settings

Kernel_types={'lin_kernel','RBF_kernel'};
Ktype=Kernel_types{1};

%%  Normal LSSVM classifier

DATA_ls={Xtr,Xval,Ytr,Yval};
DATA_2={X, Y};  % X includes Xtr and Xval.

if strcmp(Ktype,'RBF_kernel')
    disp('------------ Normal LSSVM with RBF Kernel ----------')
else
    disp('------------ Normal LSSVM with linear Kernel ----------')
end

[alpha_lssvm,b_lssvm,gamma_lssvm,sig_lssvm,minCVerr]=normal_lssvm_classifier(DATA_ls,DATA_2,Ktype,numFolds);
par_lssvm=[gamma_lssvm,sig_lssvm];
Ktest=KernelMatrix(Xtest,Ktype, sig_lssvm, X);
D=diag(Y);
pred = sign(Ktest *D* alpha_lssvm + b_lssvm);
R=reshape(pred,100,100);
figure
contourf(data.GGG,data.FFF,R)
hold on
plot(X1(:,1),X1(:,2),'ro','MarkerSize',15,'LineWidth',3)
hold on
plot(X2(:,1),X2(:,2),'b+','MarkerSize',15,'LineWidth',3)
if strcmp(Ktype,'RBF_kernel')
    title('LSSVM with RBF kernel')
else
    title('LSSVM with linear kernel')
end


%% Non-parallel LSSVM classifier 

% The tuned hyper-parameters of LSSVM is used for non-parallel lssvm classifier.
% The ratio of r=gamma_1/gamma_2 is changing in each iteration.
% one can notice when r=1 the proposed non-parallel lssvm reduces to normal lssvm.


if strcmp(Ktype,'RBF_kernel')
    disp('----------- Non-parallel LSSVM with RBF kernel ----------')
else
    disp('----------- Non-parallel LSSVM with linear kernel ----------')
end

DATA={X1, X2};
Xtr={X1tr, X2tr};
DATA_val={Xval,Yval};
ratio_range = logspace(0,4,5);  % (see Ref[1] for details)

for i=1:numel(ratio_range)
    
    r=ratio_range(i);
    [beta_1,beta_2,b_1,b_2,sig,gamma,minCVerr]=non_parallel_lssvm_classifier(DATA,Xtr,DATA_val,Ktype,r,numFolds,par_lssvm);  % if hyper par are known
    Err(i)=minCVerr;
    Par(i,:)=[sig,gamma];
    
    fprintf('ratio =%5.2f\t\t\t',r)
    fprintf('min CV error =%5.2f\n',Err(i))
    K1test_plot=KernelMatrix(Xtest,Ktype, sig, X1);
    K2test_plot=KernelMatrix(Xtest,Ktype, sig, X2);
    y1 = K1test_plot* beta_1{1} - K2test_plot*beta_1{2} + b_1;
    y2 = K2test_plot* beta_2{1} + K1test_plot*beta_2{2} + b_2;
    Dist=[abs(y1) , abs(y2)];
    [II,CC] = min(Dist,[],2);
    RR=reshape(CC,100,100);
    figure
    hold on
    contourf(data.GGG,data.FFF,RR)
    hold on
    plot(X1(:,1),X1(:,2),'ro','MarkerSize',15,'LineWidth',3)
    hold all
    plot(X2(:,1),X2(:,2),'b+','MarkerSize',15,'LineWidth',3)
    
    if strcmp(Ktype,'RBF_kernel')
        title(['r=',num2str(r),', non-parallel LSSVM with RBF Kernel'])
    else
        title(['r=',num2str(r),', non-parallel LSSVM with linear Kernel'])
    end
    pause(1)
    
end