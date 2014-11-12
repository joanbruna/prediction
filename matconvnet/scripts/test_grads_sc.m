
range = 1;

X = range* rand(1,1,50,5);
%X = single(X);

D = range*randn(50,20);
D = mexNormalize(D);
%D = single(D);

lambda = 0.2;

Z = vl_sc(X,D,lambda);


Z0 = rand(size(Z));

obj = 0.5*norm(Z(:)-Z0(:),'fro')^2;
dzdy = Z-Z0;

[dzdx,dzdw] = vl_sc(X,D,lambda,dzdy,'single',0);

epsilon = 1e-4;

dX = epsilon*randn(size(X));
X_ = X + dX;

Z_ = vl_sc(X_,D,lambda,[],'single',0);

%y = vl_nnconv(x,w,b,'verbose') ;


obj_ = 0.5*norm(Z_(:)-Z0(:),'fro')^2;


[obj_-obj, dzdx(:)'*dX(:)]/epsilon

% dictionary

epsilon = 1e-4;

dD = zeros(size(D));
dD = epsilon*randn(size(D));
%dD_ = epsilon*randn(size(D));
%dD(33,12) = dD_(33,12);
D_ = D + dD;

Z_ = vl_sc(X,D_,lambda,[],'single',0);


obj_ = 0.5*norm(Z_(:)-Z0(:),'fro')^2;

[obj_-obj, dzdw(:)'*dD(:)]/epsilon


%% Construct the network

X = single(X);
D = single(D);

f =1;
net_sc.layers = {} ;
net_sc.layers{end+1} = struct('type', 'sc', ...
                           'dict', D,...
                           'lambda', lambda, ...
                           'stride', 1, ...
                           'pad', 0) ;
net_sc.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,size(D,2),10, 'single'),...
                           'biases', zeros(1,10,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net_sc.layers{end+1} = struct('type', 'softmaxloss') ;

%% Run forward pass

X = single(X);

net_sc.layers{end}.class = single(ones(1,size(X,4)));

res = [];
res = vl_simplenn(net_sc, X, single(1), res, ...
    'disableDropout', true, ...
    'conserveMemory', 1, ...
    'sync', 1) ;


%% B-prop

epsilon = 1e-4;

res = [];
res_bp = vl_simplenn(net_sc, X, single(1), res, ...
    'disableDropout', true, ...
    'conserveMemory', 1, ...
    'sync', 1) ;

net_a = net_sc;
net_b = net_sc;

i = 33;
j = 11;

net_a.layers{1}.dict(i,j)     =  net_a.layers{1}.dict(i,j)  + epsilon;
net_b.layers{1}.dict(i,j)     =  net_b.layers{1}.dict(i,j)  - epsilon;

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