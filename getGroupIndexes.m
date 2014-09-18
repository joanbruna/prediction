function [indexes,indexes_inv] = getGroupIndexes(K,M,Gf,Gt,v)


gsize = Gf*Gt;

Cind = reshape(1:K*M,K,M);
Mind = im2col(Cind,[Gf,Gt],'distinct');
Mind_inv = reshape(invperm(Mind),K,M);


% Number of shifts
S = size(v,1);

indexes = cell(1,S);
indexes_inv = cell(1,S);

indexes{1} = Mind;
indexes_inv{1} = Mind_inv;

if S>1
for i=2:S
    
    Cind_c = circshift(Cind, [v(i,1) v(i,2)]);
    Mind = im2col(Cind_c,[Gf,Gt],'distinct');
    Mind_inv = reshape(invperm(Mind),K,M);
    
    indexes_inv{i} = Mind_inv;
    indexes{i} = Mind;
    
end
end
