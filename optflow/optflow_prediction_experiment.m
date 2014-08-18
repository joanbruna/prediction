close all;
clear all;

addpath ..
addpath ../grouplasso

load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/dictionary_s4_sort.mat');
load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s4.mat');

Xqn = Xc(:,1:500);
Xqn = mexNormalize(Xqn);


p.sigma = 0.1;
p.hn = 5;
p.lambda = 0.1;
p.lambdar = 0.1;

gp.nmf = 1;
gp.overlapping=1;
gp.alpha_iters = 200;
gp.v = [0,0;1,0;0,1;1,1];
gp.lambda = 0.1;

options.mu = 1;

niter = 10;

X0=Xqn(:,1:200);

K=size(DD,2);
M=size(X0,2);

Agl = time_coeffs_update(DD, X0, gp);
rec = DD*Agl;

norm(rec(:)-X0(:))/norm(X0(:))

keyboard;


tic;A0 = nmf_optflow( X0, DD, zeros(K,M), options);toc
A = A0;


[theta,estim] = optflow_taylor2(A0, p,zeros(K,M));

theta0 = theta;


for i = 1:niter

   tic;[A,c(i)] = nmf_optflow( X0, DD, theta, options,A);toc

   
   [theta,estim] = optflow_taylor2(A, p,theta);
   
   
%   b(i+1) = norm(A(:,2:end)-estim(:,1:end-1),'fro');
end

figure(1)
subplot(211)
imagesc(A0)
subplot(212)
imagesc(theta0)

figure(2)
subplot(211)
imagesc(A)
subplot(212)
imagesc(theta)

figure(3)
plot(c,'r')


