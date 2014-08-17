
p.sigma = 0.1;
p.hn = 5;
p.lambda = 0.1;
p.lambdar = 0.1;

options.mu = 1;

niter = 10;

A0 = nmf_optflow( Xqn, DD, zeros(K,M), options);
A = A0;


[theta,estim] = optflow_taylor2(A0, p,zeros(K,M));

theta0 = theta;


for i = 1:niter

   [A,c(i)] = nmf_optflow( Xqn, DD, theta, options,A);

   
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
