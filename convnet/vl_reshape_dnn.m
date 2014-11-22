function varargout = vl_reshape_dnn(X,N,C,dzdy,varargin)
%    
%

if nargin <= 3
    dzdy = [];
end


% no division by zero
%X = X + 1e-4 ;


%---------------------------------
% Parameters
opts.output = 'Y';
opts = vl_argparse(opts, varargin);
%---------------------------------

% reshape to use mexLasso function
sz = size(X);


if isempty(dzdy)
    
    X1 = permute( X(:,:,1:sz(3)/2,:), [3,2,1,4]);
    X2 = permute( X(:,:,(sz(3)/2+1):end,:), [3,2,1,4]);
    
    X1 = reshape( X1, [N,C,1,size(X1,4)]);
    X2 = reshape( X2, [N,C,1,size(X2,4)]);
    
%     Y(:,:,1,:) = X1;
%     Y(:,:,2,:) = X2;
    %varargout{1} = Y;
    varargout{1} = cat(3,X1,X2);
    
    
else 
    
    aux = reshape( dzdy, [N*C,1,2,size(dzdy,4)]);
    Y1 = permute( aux(:,:,1,:), [3,2,1,4]);
    Y2 = permute( aux(:,:,2,:), [3,2,1,4]);
    varargout{1} = cat(3,Y1,Y2);
    
end