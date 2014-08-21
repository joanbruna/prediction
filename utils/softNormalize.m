function [X,norms] = softNormalize(X,epsilon)

if ~exist('epsilon','var')
    epsilon = 0.5;
end

norms = sqrt(sum(X.^2));
X = X ./ repmat(sqrt(epsilon^2+sum(X.^2)),size(X,1),1) ;