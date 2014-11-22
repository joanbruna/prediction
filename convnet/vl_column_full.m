function varargout = vl_column_full(X,dzdy,varargin)
%    
%

if nargin <= 4
    dzdy = [];
end


% no division by zero
%X = X + 1e-4 ;


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

if numel(sz)<3
    sz(3) = 1;
    sz(4) = 1;
elseif numel(sz)<4
    sz(4) = 1;
end

X = reshape(X,[sz(1)*sz(3),sz(2),1,sz(4)]);


out = vl_nnconv(X, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride) ;




