function D=TrainDL_wrapper(X, param)
%this function calls mexTrainDL repeated times, so that we can track its evolution

param0 = param;

verb = getoptions(param,'verb', 500);
iter = getoptions(param,'iter', 5000);
param0.iter = verb;
L=round(iter/verb);


I=randperm(size(X,2));
validset=getoptions(param,'evalset',1000);
Xeval=X(:,I(1:validset));

param0

for l=1:L
fprintf('running iters from %d to %d...', 1+(l-1)*verb, l*verb)
%run MexTrainDL
D = mexTrainDL(X, param0);
alpha = mexLasso(Xeval, D, param0);
rec = D * alpha;
param0.D = D;

c1 = full(.5*norm(Xeval(:)-rec(:))^2)/verb;
c2 = full(param0.lambda*sum(abs(alpha(:))))/verb;

fprintf('cost is %f (%f, %f) \n', c1+c2,c1,c2)

end


