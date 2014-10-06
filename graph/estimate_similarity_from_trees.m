function S=estimate_similarity_from_trees(T,sigma)

if nargin < 2
sigma = 4;
end

N=length(T{1});
S=zeros(N);
ntrees=size(T,2);

for t=1:ntrees
S = S + tree_similarity(T{t},N,sigma)/ntrees;

end


end


function out=tree_similarity(T,N,sigma)

out = abs(repmat(T,1,N) - repmat(T',N,1));
out = exp(-out.^2/sigma);

end
