close all;
clear all;

%addpath ..
%addpath ../grouplasso

load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/dictionary_s4_sort.mat');
load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s4.mat');

Xqn = Xc(:,1:500);
Xqn = mexNormalize(Xqn);


p.sigma = 1;
p.hn = 5;
p.lambda = 0.1;
p.lambdar = 0.1;

gp.nmf = 1;
gp.overlapping=1;
gp.alpha_iters = 200;
gp.v = [0,0;1,0;0,1;1,1];
gp.lambda = 0.2
gp.groupsize=1;
gp.time_groupsize=1;

options.mu = 1;

niter = 10;

X0=Xqn(:,1:400);

K=size(DD,2);
M=size(X0,2);

tic
Agl = time_coeffs_update(DD, X0, gp);
toc
rec = DD*Agl;

norm(rec(:)-X0(:))/norm(X0(:))

%keyboard;

A0 = nmf_optflow( X0, DD, zeros(K,M), p);
rec = DD*A0;
norm(rec(:)-X0(:))/norm(X0(:))

%tic;A0 = nmf_optflow( X0, DD, zeros(K,M), options);toc
%A = A0;

[theta,estim,estimfut] = optflow_taylor2(Agl, p, 0);

%keyboard;

%check estim is doing what it should
Apast=0*Agl;
Apast(:,2:end)=Agl(:,1:end-1);
Afut=0*Agl;
Afut(:,1:end-1)=Agl(:,2:end);


relpast = eps+sqrt(sum(Apast.^2));
relpres = eps+sqrt(sum(Agl.^2));
relfut = eps+sqrt(sum(Afut.^2));



norm(estim(:)-Apast(:))/norm(Apast(:))
norm(estim(:)-Agl(:))/norm(Agl(:))
norm(estim(:)-Afut(:))/norm(Afut(:))





