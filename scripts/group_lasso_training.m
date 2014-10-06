close all
clear all 

addpath utils
addpath stft
addpath ../video_prediction
addpath grouplasso

%this script tries a group lasso training. 

epsilon = 1;
tmp = load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/joint.mat');

X = tmp.X ./ repmat(sqrt(epsilon^2+sum(tmp.X.^2)),size(tmp.X,1),1) ;
Xt_same = tmp.Xt_same ./ repmat(sqrt(epsilon^2+sum(tmp.Xt_same.^2)),size(tmp.X,1),1) ;
Xt_different = tmp.Xt_different ./ repmat(sqrt(epsilon^2+sum(tmp.Xt_different.^2)),size(tmp.X,1),1) ;


%%%try first logarithmic scale and standard group lasso 
X=log(X+eps);

mu=mean(X,2);
Xc=X-repmat(mu,1,size(X,2));

options.renorm_input = 1;
options.K=800;
options.lambda=0.05;
options.time_groupsize = 2;
options.groupsize=2;
options.batchsize = 1024;
options.epochs=1;
options.overlapping=1;
options.alpha_iters=200;

if 1
[D, D0, verbo] = group_pooling_st(Xc, options);
else
tutu=load('/home/bruna/matlab/prediction/grouplassoDict.mat');
D=tutu.D;
end

keyboard; 
%% prediction test easy %%%

% 1 obtain the synthesis coefficients
%[~,~,alphas] = time_coeffs_update(D, Xc, options);
chsz=2^16;
nch=floor(size(Xc,2)/chsz);

alphas=zeros(size(D,2),size(Xc,2));
for n=1:nch
[~,~,alphas(:,1+(n-1)*chsz:n*chsz)]=time_coeffs_update22(D, Xc(:,1+(n-1)*chsz:n*chsz), options);
fprintf('done %d examples \n', n*chsz)

end


keyboard;

%2 linear regression on pooled alphas to obtain the optimal linear dynamics
[modulus,phase] = modphas_decomp(alphas,options.groupsize);
[mlin_pred_err, Am , mpred] = linear_prediction(modulus, [3:size(alphas,2)], 1);

%sanity check: verify that it is much easier to predict modulus than raw coeffs
[alin_pred_err, Aa , alpha_pred] = linear_prediction(alphas, [3:size(alphas,2)], 1);

%this is not what we want: we want just scalar regressions, but let's see...
%[plin_pred_err, Ap , ppred] = linear_prediction(phase, [3:size(alphas,2)], 1);

%3 phase predictor: 




