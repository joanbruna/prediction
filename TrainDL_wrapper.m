function D=TrainDL_wrapper(X, param)
%this function calls mexTrainDL repeated times, so that we can track its evolution

addpath nmf_linear_dynamics
param0 = param;

verb = getoptions(param,'verb', 200);
iter = getoptions(param,'iter', 2000);
param0.iter = verb;
L=round(iter/verb);


I=randperm(size(X,2));
validset=getoptions(param,'evalset',100);
Xeval=X(:,I(1:validset));

param0

for l=1:L
fprintf('running iters from %d to %d...', 1+(l-1)*verb, l*verb)
%run MexTrainDL
D = mexTrainDL(X, param0);
alpha = mexLasso(Xeval, D, param0);
alphap = nmf_linear_dynamic_pursuit(Xeval, D, eye(size(D,2)), param0);
rec = D * alpha;
param0.D = D;

c1 = full(.5*norm(Xeval(:)-rec(:))^2)/validset;
c2 = full(param0.lambda*sum(abs(alpha(:))))/validset;
c2p = full(param0.lambda*sum(abs(alphap(:))))/validset;

fprintf('cost is %f (%f, %f, %f) \n', c1+c2,c1,c2,c2p)

end


