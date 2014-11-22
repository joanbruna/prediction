function varargout = vl_fit(X,Ymix,Y1,Y2,dzdy,varargin)
%    Y = VL_FILTERMASK(X, F, B) 
%
%
%


if nargin <= 4
    dzdy = [];
end

% no division by zero


%---------------------------------
% Parameters
opts.loss = 'L2';
opts = vl_argparse(opts, varargin);
%---------------------------------

n_frames = size(X,2);

switch opts.loss
    case 'L2'
        
        X1 = repmat(X(:,:,1,:),[1,1,2,1]);
        X2 = repmat(X(:,:,2,:),[1,1,2,1]);     
        
        if isempty(dzdy)

        varargout{1} = 0.5*sum((Ymix(:).*X1(:) - Y1(:)).^2)/n_frames + 0.5*sum((Ymix(:).*X2(:) - Y2(:)).^2)/n_frames;
        
        else
            aux1 = Ymix(:,:,1,:).*( Ymix(:,:,1,:).*X(:,:,1,:) - Y1(:,:,1,:))/n_frames + Ymix(:,:,2,:).*(Ymix(:,:,2,:).*X(:,:,1,:) - Y1(:,:,2,:))/n_frames ;
            aux2 = Ymix(:,:,1,:).*( Ymix(:,:,1,:).*X(:,:,2,:) - Y2(:,:,1,:))/n_frames + Ymix(:,:,2,:).*(Ymix(:,:,2,:).*X(:,:,2,:) - Y2(:,:,2,:))/n_frames ;
            
            varargout{1} = cat(3,aux1,aux2);
            
        end
        
        
        
        
        
end

