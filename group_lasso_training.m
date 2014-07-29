close all
clear all 

addpath utils
addpath stft

%this script tries a group lasso training. 

numspeakers=20;
samples_speaker=10000;

X=zeros(321,numspeakers*samples_speaker);
for i=1:numspeakers
tmp = load(sprintf('/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s%d.mat',i));
X(:,1+(i-1)*samples_speaker:i*samples_speaker)=tmp.Xc(:,1:samples_speaker);
i
end


%%%try first logarithmic scale and standard group lasso 
X=log(X+eps);

options.renorm_input = 1;
options.K=400;
options.lambda=0.05;
options.time_groupsize = 2;
options.batchsize = 2000;

[D, D0, verbo] = group_pooling_st(X, options);



