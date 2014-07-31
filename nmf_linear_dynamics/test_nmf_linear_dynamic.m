

% param.K=128; % learns a dictionary with 100 elements 
% param.lambda=0.05; 
% param.numThreads=12;	%	number	of	threads 
% param.batchsize =128;
% param.iter=500; % let us see what happens after 1000 iterations .
% param.posD=1;
% param.posAlpha=1;
% param.pos=1;
% 
% Dini=mexTrainDL(X, param);

addpath nmf_linear_dynamics/
addpath utils
addpath ../spams-matlab/build/

%X = mexNormalize(X);

clear param

param = struct;
%param.D = Dini;
param.K = 1000;
param.lambda = 0.05;
param.mu = 10;
param.epochs = 2;
param.batchsize = 100;
param.renorm_input = 1;


[D,W,verbo] = nmf_linear_dynamic(X, param);


%%

Xt = mexNormalize(Xt_same);



%%

n=200;
temp_sup = 2;
idx = randperm(size(Xt,2));

for i=1:n

   chunk = Xt(:,idx(i):idx(i)+temp_sup);
   alpha = nmf_linear_dynamic_pursuit( chunk, D, W , param);
    
   r(i) = sum( (Xt(:,idx(i)+temp_sup+1) - D*W*alpha(:,end)).^2 );
   
   m(i) = sum( (Xt(:,idx(i)+temp_sup+1) - Xt(:,idx(i)+temp_sup)).^2);
   
end
[sqrt(mean(r)) sqrt(mean(m))]
















