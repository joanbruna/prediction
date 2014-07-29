function out=ProximalFlat(in, I0, I1, lambda,G,M)

aux=reshape(in(I0,:),G,M);
normes=repmat(sqrt(sum(aux.^2)),[G 1]);
I=find(normes>0);
aux(I) = aux(I).*(max(0,normes(I)-lambda)./normes(I));
aux = reshape(aux, size(in));
out= aux(I1,:);


