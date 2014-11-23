function [X,norms] = softNormalize(X,epsilon,channel)

if ~exist('epsilon','var')
    epsilon = 0.5;
end

if ~exist('channel','var')
    channel = 1;
end


norms = sqrt(sum(abs(X).^2, channel) );
if channel==3
X = X ./ repmat(sqrt(epsilon^2+sum(abs(X).^2,3)),1,1,size(X,3),1) ;
else
X = X ./ repmat(sqrt(epsilon^2+sum(abs(X).^2)),size(X,1),1) ;
end
%X = X ./ repmat(sqrt(epsilon^2+sum(abs(X).^2)),size(X,1),1) ;
