function [indexes,indexes_inv] = getTreeIndexes(K,M,trees,Gt, Jmax)


if mod(M,2)>0
error('we do not currently deal with odd batchsizes')
end

%assume temporal overlapping by default
ntrees=size(trees,2);

for t=1:ntrees

Cind= repmat(trees{t},1,M)+K*repmat([0:M-1],K,1);
for j=1:Jmax
indexes{2*t-1}{j} = im2col(Cind,[2^j,Gt],'distinct');
indexes_inv{2*t-1}{j} = reshape(invperm(indexes{2*t-1}{j}),K,M);
indexes{2*t-0}{j} = im2col(circshift(Cind,[0 Gt/2]) ,[2^j,Gt],'distinct');
indexes_inv{2*t-0}{j} = reshape(invperm(indexes{2*t}{j}),K,M);
end
end


