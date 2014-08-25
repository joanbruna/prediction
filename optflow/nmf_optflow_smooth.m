
function  [A,theta,SA,Z] = nmf_optflow_smooth(X,D,options,ptheta)


if ~exist('ptheta','var')
    ptheta = struct;
    ptheta.sigma = 1;
    ptheta.hn = 11;
    ptheta.lambda = 0.1;
    ptheta.lambdar = 0.00001;
end

options.lambda_t = ptheta.lambda;
options.lambda_tr = ptheta.lambdar;
options.hn = ptheta.hn;
options.sigma = ptheta.sigma;


K = size(D,2);
M = size(X,2);


[A0,~,SA,Z] = nmf_optflow( X, D, zeros(K,M), options);
A = A0;


theta = optflow_taylor2(A0, ptheta,zeros(K,M));
%[theta,estim] = optflow_taylor_temp(A0, ptheta);

theta0 = theta;

total_iter=getoptions(options,'total_iter',5);

c = zeros(1,total_iter);


for i = 1:total_iter
    
   [A,c(i),SA,Z] = nmf_optflow( X, D, theta, options,[A;Z]);
   
   %[theta,estim] = optflow_taylor2(A, ptheta,theta);
   theta = optflow_taylor_temp(A, ptheta);
   
end


pt=getoptions(options,'pt',0);

if pt
subplot(311)
dbimagesc(X+0.001);
subplot(312)
imagesc(A)
subplot(313)
imagesc(SA)
end
