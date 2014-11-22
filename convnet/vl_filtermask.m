function varargout = vl_filtermask(X,p,dzdy,varargin)
%    Y = VL_FILTERMASK(X, F, B) 
%
%
%

if nargin <= 1
    p = 2;
end

if nargin <= 2
    dzdy = [];
end

% no division by zero


%---------------------------------
% Parameters
opts.output = 'Y';
opts = vl_argparse(opts, varargin);
%---------------------------------

eps = 1e-6;

DEN = X(:,:,1,:).^p+X(:,:,2,:).^p + eps; 

Z = 0*X;%zeros(size(X));

if isempty(dzdy)
    Z(:,:,1,:) = (X(:,:,1,:).^p)./DEN; 
    Z(:,:,2,:) = (X(:,:,2,:).^p)./DEN;
else 
    GM = ( p*X(:,:,1,:).^(p-1).*( X(:,:,2,:).^p + eps)./(DEN.^2 ));
    Z(:,:,1,:) =  dzdy(:,:,1,:).*GM - dzdy(:,:,2,:).*GM;
    
    GM = ( p*X(:,:,2,:).^(p-1).*( X(:,:,1,:).^p + eps)./(DEN.^2 ));
    Z(:,:,2,:) =  -dzdy(:,:,1,:).*GM + dzdy(:,:,2,:).*GM;
end

varargout{1} = Z;



%2*((W1H1).*(W2H2.^2 + const))./( ( W1H1.^2 + W2H2.^2 + const).^2 );



