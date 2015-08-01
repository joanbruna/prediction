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
opts.find_match = 0;
opts = vl_argparse(opts, varargin);
find_match = opts.find_match;
%---------------------------------
    
switch opts.loss
    
    case 'L2'
        
        D1 = Ymix.*X(:,:,1:2:end,:) - Y1;
        D2 = Ymix.*X(:,:,2:2:end,:) - Y2;
        
        cost = 0.5*sum(abs(D1(:)).^2) + .5 *sum(abs(D2(:)).^2);
        
        
        if isempty(dzdy)
            
            varargout{1} = cost;
            
        else
            
            Df = zeros(size(X),'single','gpuArray');
            
            Df(:,:,1:2:end,:) = real(conj(Ymix).*D1);
            Df(:,:,2:2:end,:) = real(conj(Ymix).*D2);
            varargout{1} = Df;
        end
        
    case 'L2_center'
        
        tc = size(X,2);
        D1 = Ymix.*X(:,ceil(tc/2),1:2:end,:) - Y1;
        D2 = Ymix.*X(:,ceil(tc/2),2:2:end,:) - Y2;
        
        cost = 0.5*sum(abs(D1(:)).^2) + .5 *sum(abs(D2(:)).^2);
        
        
        if isempty(dzdy)
            
            varargout{1} = cost / (.5*sum(Y1(:).^2) + .5*sum(Y2(:).^2));
            
        else
            
            Df = zeros(size(X),'single','gpuArray');
            
            Df(:,ceil(tc/2),1:2:end,:) = real(conj(Ymix).*D1);
            Df(:,ceil(tc/2),2:2:end,:) = real(conj(Ymix).*D2);
            varargout{1} = Df;
        end
        
end




    
% old

% % Parameters
% opts.loss = 'L2';
% opts.find_match = 0;
% opts = vl_argparse(opts, varargin);
% find_match = opts.find_match;
% %---------------------------------
% 
% %keyboard
% 
% 
% if ~find_match
%     
%     switch opts.loss
%     
%     case 'L2'
%    
%         D1 = Ymix.*X(:,:,1:2:end,:) - Y1;
%         D2 = Ymix.*X(:,:,2:2:end,:) - Y2;
%         
%         cost = 0.5*sum(abs(D1(:)).^2) + .5 *sum(abs(D2(:)).^2);
%     
%     case 'center'
%     
%     
%     end
%     
% else
%     
%     aux_1 = reshape( 0.5*sum(sum(abs(D1).^2,3),2) + .5 *sum(sum(abs(D2).^2,3),2), [1,size(X,4)] );
%     
%     D1_ = Ymix.*X(:,:,2:2:end,:) - Y1;
%     D2_ = Ymix.*X(:,:,1:2:end,:) - Y2;
%     
%     aux_2 = reshape( 0.5*sum(sum(abs(D1_).^2,3),2) + .5 *sum(sum(abs(D2_).^2,3),2), [1,size(X,4)] );
%     
%     [m,idx] = min([aux_1;aux_2]);
%     
%     cost = sum(m);
%     
% end
% 
% if isempty(dzdy)
%     
%     varargout{1} = cost;
%     
% else
%     
%     Df = zeros(size(X),'single','gpuArray');
%     
%     if find_match
% 
%         D1(:,:,:,idx==2) =  D1_(:,:,:,idx==2);
%         D2(:,:,:,idx==2) =  D2_(:,:,:,idx==2);
%         
%     end
%     Df(:,:,1:2:end,:) = real(conj(Ymix).*D1);
%     Df(:,:,2:2:end,:) = real(conj(Ymix).*D2);
%     varargout{1} = Df;
% end

