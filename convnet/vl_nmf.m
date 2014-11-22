function varargout = vl_nmf(X,D1,D2,lambda,dzdy,varargin)
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

if nargin <= 4
    dzdy = [];
end


% no division by zero
%X = X + 1e-4 ;


k1 = size(D1,2);
k2 = size(D2,2);

%---------------------------------
% Parameters
opts.output = 'Y';
opts.pos = 0;
opts.single = 1;
opts.lambda2 = 0.001;
opts = vl_argparse(opts, varargin);
%---------------------------------

% reshape to use mexLasso function
sz = size(X);
X = permute(X,[1,3,2,4]);
sv = size(X);
if numel(sz)<3
    sz(3) = 1;
    sz(4) = 1;
elseif numel(sz)<4
    sz(4) = 1;
end

Xv = reshape(X,[sz(1)*sz(3),sz(2)*sz(4)]);

% Define sparse coding parameters
lasso_params.pos = 1;
lasso_params.lambda = lambda;
lasso_params.lambda2 = opts.lambda2;
lasso_params.pos = opts.pos;


Z = full(mexLasso(Xv,[D1,D2],lasso_params));


Z1 = single(Z(1:k1,:));
Z2 = single(Z(k1+1:end,:));


if isempty(dzdy)

        
    Y = zeros(sz(1)*sz(3),sz(2),1,sz(4));
    Y(:,:,1,:) = reshape(D1*Z1,[sz(1)*sz(3),sz(2),1,sz(4)]);
    Y(:,:,2,:) = reshape(D2*Z2,[sz(1)*sz(3),sz(2),1,sz(4)]);
    
    varargout{1} = Y;
    
else
    M = eye(sz(3));
    
    dzdy1v = reshape(dzdy(:,:,1,:),[sz(1)*sz(3),sz(2)*sz(4)]); 
    dzdy2v = reshape(dzdy(:,:,2,:),[sz(1)*sz(3),sz(2)*sz(4)]);  
    
    dzdyv = [D1'*dzdy1v;D2'*dzdy2v];
    
    [dD, dzdx_aux] = lasso_grads(gather(Xv), gather(Z), gather(dzdyv), gather(M), gather([D1,D2]), lasso_params.lambda2);

    
    dzdx_aux = reshape(dzdx_aux,[sz(1),sz(3),sz(2),sz(4)]);

    varargout{1} = permute(dzdx_aux,[1,3,2,4]);

    %dzdx = zeros(sz);
    
    dwdz{1} = dD(:,1:k1) + dzdy1v*Z1';
    dwdz{2} = dD(:,k1+1:end) + dzdy2v*Z2';
    
    varargout{2} = dwdz;
    
    
    
    
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

function Xv = vectorize(X)

X = permute(X,[1,3,2,4]);
sz = size(X);
Xv = reshape(X,[sz(1)*sz(2),sz(3)*sz(4)]);




function X = inv_vectorize(Xv,sz)

X = reshape(Xv,sz);
X = permute(X,[1,3,2,4]);

