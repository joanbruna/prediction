

N = 20;
M = 1000;

X = max(2 + randn(N,M),0);

param = struct;
param.K = 2*N;



[D,W,verbo] = nmf_linear_dynamic(X, param);


