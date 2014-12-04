function varargout = vl_fit(X,Ymix,Y1,Y2,dzdy,varargin)
%    Y = VL_FILTERMASK(X, F, B) 
%
%
%

%X is of size [1 x K x 2N x BS]
%Yi is of size [1 x K x N x BS] (complex)


if nargin <= 4
    dzdy = [];
end

% no division by zero


%---------------------------------
% Parameters
opts.loss = 'L2';
opts = vl_argparse(opts, varargin);
%---------------------------------

nframes = size(X,2);

switch opts.loss
    case 'L2'
        
	if 1
	padsize = round((size(Ymix,2) - size(X,2))/2);
	if padsize>0
		Ymix = Ymix(:,1+padsize:end-padsize,:,:);
		Y1 = Y1(:,1+padsize:end-padsize,:,:);
		Y2 = Y2(:,1+padsize:end-padsize,:,:);
	end
	end
	D1 = Ymix.*X(:,:,1:2:end,:) - Y1;
	D2 = Ymix.*X(:,:,2:2:end,:) - Y2;
        
        if isempty(dzdy)

	varargout{1} = 0.5*sum(abs(D1(:)).^2) + .5 *sum(abs(D2(:)).^2); 
        else
	    Df = zeros(size(X),'single','gpuArray');
	   Df(:,:,1:2:end,:) = real(conj(Ymix).*D1);
	   Df(:,:,2:2:end,:) = real(conj(Ymix).*D2);
	  varargout{1} = Df;
        end
        
end

