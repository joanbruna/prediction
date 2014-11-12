function varargout = vl_sc(X,D,lambda,dzdy,varargin)
%    Y = VL_NNCONV(X, F, B) computes the convolution of the image stack X
%    with the filter bank F and biases B. If B is the empty matrix,
%    then no biases are added. If F is the empty matrix, then
%    the function does not filter the image, but still adds the
%    biases as well as performing downsampling and padding as explained
%    below.
%
%    [DXDY, DXDF, DXDB] = VL_NNCONV(X, F, B, DZDY) computes the
%    derivatives of the nework output Z w.r.t. the data X and
%    parameters F, B given the derivative w.r.t the output Y. If B is
%    the empty matrix, then DXDB is also empty.

if nargin <= 3
    dzdy = [];
end


% no division by zero
%X = X + 1e-4 ;
sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;

n = sz(4);
m = size(D,1);
k = size(D,2);

%---------------------------------
% Parameters
opts.output = 'Y';
opts.pos = 0;
opts.single = 1;
opts.lambda2 = 0.001;
opts = vl_argparse(opts, varargin);
%---------------------------------

% Define sparse coding parameters
lasso_params.lambda = lambda;
lasso_params.lambda2 = opts.lambda2;
lasso_params.pos = opts.pos;

% reshape to use mexLasso function
Xv = reshape(X,[sz(3),sz(4)]);

Z = full(mexLasso(Xv,D,lasso_params));

if opts.single
    Z = single(Z);
end


if isempty(dzdy)
    if strcmp(opts.output,'DY')
        varargout{1} = reshape(D*Z,[1,1,m,n]);
    else
        varargout{1} = reshape(Z,[1,1,k,n]);
    end
else 
    M = eye(sz(3));
    
    if strcmp(opts.output,'DY')
        dzdyv_aux = reshape(dzdy,[m,sz(4)]);
        dzdyv = D'*dzdyv_aux;
    else
        dzdyv = reshape(dzdy,[k,sz(4)]);
    end
    
    [dD, dzdx_aux] = lasso_grads(gather(Xv), gather(Z), gather(dzdyv), gather(M), gather(D), lasso_params.lambda2);
    
    varargout{1} = reshape(dzdx_aux,[1,1,sz(3),sz(4)]);
    
    if strcmp(opts.output,'DY')
        
        varargout{2} = dD + dzdyv_aux*Z';
        
    else
        varargout{2} = dD;
    end
    
    
end


function [dD,dY] = lasso_grads(X, Y, dZdY, M, D, lambda2)

if size(X,2) > 1
    dD = zeros(size(D));
    dY = zeros(size(X));
    for k=1:size(X,2)
        [dD_,dY_ ] = lasso_grads(X(:,k), Y(:,k), dZdY(:,k), M, D, lambda2);
        dD = dD + dD_;
        dY(:,k) = dY_;
    end
    dY = single(dY);
    dD = single(dD);
    return
end

% Find active set
id = find(Y ~=0);
lact = length(id);

Dd = D(:,id);

b = zeros(size(Y));
b(id) = (Dd'*Dd + lambda2*eye(lact))\dZdY(id);

% Gradient
dD = -D*b*Y'+(X-D*Y)*b';

B = (Dd'*Dd + lambda2*eye(lact))\Dd';

dY = B'*dZdY(id);

