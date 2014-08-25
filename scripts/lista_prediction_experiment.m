close all
clear all 

addpath utils
addpath stft
addpath grouplasso
addpath('../video_prediction')

tmp = load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/joint.mat');

X = tmp.X ./ repmat(sqrt(sum(tmp.X.^2)),size(tmp.X,1),1) ;
Xt_same = tmp.Xt_same ./ repmat(sqrt(sum(tmp.Xt_same.^2)),size(tmp.X,1),1) ;
Xt_different = tmp.Xt_different ./ repmat(sqrt(sum(tmp.Xt_different.^2)),size(tmp.X,1),1) ;


[N,L]=size(X);

[lin_pred_err, A , Xpred] = linear_prediction(X, [3:size(tmp.X,2)], 1);

X = X - Xpred ;

X = mexNormalize(X) ;

X1=X(:,1:end-2);
X2=X(:,2:end-1);
Y=X(:,3:end);

X0=[X1 ; X2];
clear X1; 
clear X2;

I=randperm(size(Y,2));
X0=X0(:,I);
Y=Y(:,I);
options.null=0;
options.lambda = 0.002;
options.epochs = 4;
options.lr = 1e-4;

[S, E, D, bias] = lista_regress(X0, Y, options);

keyboard;

%Xtest=X0(:,I(Ltrain+1:end));
%Ytest=Y(:,I(Ltrain+1:end));
%Ypred = lista_predict(Xtest, S, E, D, bias, options);






