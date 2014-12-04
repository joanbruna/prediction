function varargout = vl_channelPoole(X,nchannels,dzdy)
%    Y = VL_FILTERMASK(X, F, B) 
%
%
%


if nargin <= 2
    dzdy = [];
end

% no division by zero

N = size(X,4);
poole = zeros(nchannels,N,'gpuArray');
for i=1:nchannels
    poole(i,:) = reshape( sum(sum( X(1,:,i:nchannels:end,:).^2,3),2),[1,N] );
end

[~,idx] = sort(poole,'descend');

idx
if isempty(dzdy)
    Z = zeros(size(X,1),size(X,2),2*size(X,3)/nchannels,N,'gpuArray');
    for j=1:N
        Z(1,:,1:2:end,j) = X(1,:,idx(1,j):nchannels:end,j);
        Z(1,:,2:2:end,j) = X(1,:,idx(2,j):nchannels:end,j); 
    end
else 
    Z = 0*X;
    for j=1:N
        Z(:,:,idx(1,j):nchannels:end,j) = dzdy(:,:,1:2:end,j);
        Z(:,:,idx(2,j):nchannels:end,j) = dzdy(:,:,2:2:end,j); 
    end
end


varargout{1} = Z;
