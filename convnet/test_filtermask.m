% first make some data
n = 20;
m = 3;
r = 12;
p = 2;


Y = rand(n,r,2,m);
X = rand(n,r,2,m);


M = vl_filtermask(X,p);

obj = 0.5*sum( (Y(:) - M(:).*Y(:)).^2);

dzdy = (M.*Y-Y).*Y;

dzdx = vl_filtermask(X,p,dzdy);



% dH1
eps_1 = 1e-8;
dX = eps_1*randn(size(X));
X_ = X + dX;

M_ = vl_filtermask(X_,p);
obj_ = 0.5*sum( (Y(:) - M_(:).*Y(:)).^2);

[obj_-obj, dzdx(:)'*dX(:)]/eps_1

break

%%



% first make some data
n = 20;
m = 3;
r = 12;
p = 2;
f =1;

lambda = 0.15;

Y = rand(n*f,r,2,m);
X = 0.1+rand(n,r,f,m);

D1 = mexNormalize(0.1+rand(n*f,r));
D2 = mexNormalize(0.1+rand(n*f,r));


M = vl_nmf(X,D1,D2,lambda);

obj = 0.5*sum( (Y(:) - M(:)).^2);


dzdy = (M-Y);


[dzdx,dwdz] = vl_nmf(X,D1,D2,lambda,dzdy);


% dH1
eps_1 = 1e-5;
dX = eps_1*randn(size(X));
X_ = X + dX;

M_ = vl_nmf(X_,D1,D2,lambda);
obj_ = 0.5*sum( (Y(:) - M_(:)).^2);

[obj_-obj, dzdx(:)'*dX(:)]/eps_1

% DD

eps_1 = 1e-4;
dD1_ = eps_1*randn(size(D1));
dD1 = zeros(size(dD1_));
dD1(8,11) = eps_1;%*dD1_(8,11);
D1_ = D1 + dD1;

M_ = vl_nmf(X,D1_,D2,lambda);
obj_ = 0.5*sum( (Y(:) - M_(:)).^2);

aux = dwdz{1};
[obj_-obj, aux(:)'*dD1(:)]/eps_1


eps_1 = 1e-5;
dD2 = eps_1*randn(size(D2));
D2_ = D2 + dD2;

M_ = vl_nmf(X,D1,D2_,lambda);
obj_ = 0.5*sum( (Y(:) - M_(:)).^2);

aux = dwdz{2};
[obj_-obj, aux(:)'*dD2(:)]/eps_1


%%

% first make some data
n = 20;
m = 3;
r = 12;
p = 2;
f =1;

lambda = 0.15;

Y = rand(n*f,r,2,m);
Y1 = rand(n*f,r,2,m);
Y2 = rand(n*f,r,2,m);
X = 0.1+rand(n,r,2,m);

% idx = 1;
% X(:,:,idx,:) = 0;

obj = vl_fit(X,Y,Y1,Y2,[],'loss','L2');


dzdx = vl_fit(X,Y,Y1,Y2,1,'loss','L2');


% dH1
eps_1 = 1e-7;
dX = eps_1*randn(size(X));
% dX(:,:,idx,:) = 0;
X_ = X + dX;

obj_ = vl_fit(X_,Y,Y1,Y2,[],'loss','L2');


[obj_-obj, dzdx(:)'*dX(:)]/eps_1





%% Test net



% first make some data
n = 20;
m = 3;
r = 12;
p = 2;
f =1;

lambda = 0.15;

Y = rand(n*f,r,2,m);
Y1 = rand(n*f,r,2,m);
Y2 = rand(n*f,r,2,m);
X = 0.1+rand(n,r,f,m);

X = single(X);


D1 = mexNormalize(0.1+rand(n*f,r));
D2 = mexNormalize(0.1+rand(n*f,r));



net_nmf.layers = {} ;
net_nmf.layers{end+1} = struct('type', 'nmf', ...
                           'D1', single(D1), ...
                           'D2', single(D2), ...
                           'lambda',lambda,...
                           'stride', 1, ...
                           'pad', 0) ;

net_nmf.layers{end+1} = struct('type', 'fitting', ...
                           'loss', 'L2') ;


net_nmf.layers{end}.Ymix = Y;
net_nmf.layers{end}.Y1 = Y1;
net_nmf.layers{end}.Y2 = Y2;



epsilon = 1e-4;

res = [];
res_bp = vl_simplenn(net_nmf, X, single(1), res, ...
    'disableDropout', true, ...
    'conserveMemory', 0, ...
    'sync', 1) ;

net_a = net_nmf;
net_b = net_nmf;

ii = randperm(size(D1,1));
jj = randperm(size(D1,2));
i = ii(1);
j = jj(1);

net_a.layers{1}.D1(i,j)     =  net_a.layers{1}.D1(i,j)  + epsilon;
net_b.layers{1}.D1(i,j)     =  net_b.layers{1}.D1(i,j)  - epsilon;

res_a = vl_simplenn(net_a, X, [], res, ...
    'disableDropout', true, ...
    'conserveMemory', 1, ...
    'sync', 1) ;

obj_a = res_a(end).x;

res_b = vl_simplenn(net_b, X, [], res, ...
    'disableDropout', true, ...
    'conserveMemory', 1, ...
    'sync', 1) ;

obj_b = res_b(end).x;
grad_num = (obj_a-obj_b)/(2*epsilon);

grad = res_bp(1).dzdw{1}(i,j);

[ grad_num, grad, abs(grad - grad_num)/abs(grad)*100]
                       
          
net_a = net_nmf;
net_b = net_nmf;

ii = randperm(size(D2,1));
jj = randperm(size(D2,2));
i = ii(1);
j = jj(1);

net_a.layers{1}.D2(i,j)     =  net_a.layers{1}.D2(i,j)  + epsilon;
net_b.layers{1}.D2(i,j)     =  net_b.layers{1}.D2(i,j)  - epsilon;

res_a = vl_simplenn(net_a, X, [], res, ...
    'disableDropout', true, ...
    'conserveMemory', 1, ...
    'sync', 1) ;

obj_a = res_a(end).x;

res_b = vl_simplenn(net_b, X, [], res, ...
    'disableDropout', true, ...
    'conserveMemory', 1, ...
    'sync', 1) ;

obj_b = res_b(end).x;
grad_num = (obj_a-obj_b)/(2*epsilon);

grad = res_bp(1).dzdw{2}(i,j);

[ grad_num, grad, abs(grad - grad_num)/abs(grad)*100]
                       
 %%
 
 % first make some data
n = 20;
m = 3;
r = 12;
p = 2;
f =1;

lambda = 0.15;

Y = rand(n*f,r,2,m);
Y1 = rand(n*f,r,2,m);
Y2 = rand(n*f,r,2,m);
X = 0.1+rand(n,r,f,m);

X = single(X);


D1 = mexNormalize(0.1+rand(n*f,r));
D2 = mexNormalize(0.1+rand(n*f,r));



net_nmf.layers = {} ;
net_nmf.layers{end+1} = struct('type', 'nmf', ...
                           'D1', single(D1), ...
                           'D2', single(D2), ...
                           'lambda',lambda,...
                           'stride', 1, ...
                           'pad', 0) ;
                       
net_nmf.layers{end+1} = struct('type', 'filtermask', ...
                           'p',2) ;

net_nmf.layers{end+1} = struct('type', 'fitting', ...
                           'loss', 'L2') ;


net_nmf.layers{end}.Ymix = Y;
net_nmf.layers{end}.Y1 = Y1;
net_nmf.layers{end}.Y2 = Y2;



epsilon = 1e-4;

res = [];
res_bp = vl_simplenn(net_nmf, X, single(1), res, ...
    'disableDropout', true, ...
    'conserveMemory', 0, ...
    'sync', 1) ;

net_a = net_nmf;
net_b = net_nmf;

ii = randperm(size(D2,1));
jj = randperm(size(D2,2));
i = ii(1);
j = jj(1);

net_a.layers{1}.D1(i,j)     =  net_a.layers{1}.D1(i,j)  + epsilon;
net_b.layers{1}.D1(i,j)     =  net_b.layers{1}.D1(i,j)  - epsilon;

res_a = vl_simplenn(net_a, X, [], res, ...
    'disableDropout', true, ...
    'conserveMemory', 1, ...
    'sync', 1) ;

obj_a = res_a(end).x;

res_b = vl_simplenn(net_b, X, [], res, ...
    'disableDropout', true, ...
    'conserveMemory', 1, ...
    'sync', 1) ;

obj_b = res_b(end).x;
grad_num = (obj_a-obj_b)/(2*epsilon);

grad = res_bp(1).dzdw{1}(i,j);

[ grad_num, grad, abs(grad - grad_num)/abs(grad)*100]
                       
                       




