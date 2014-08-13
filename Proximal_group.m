function out = Proximal_group(X,indexes,indexes_inv, lambda)


S = length(indexes);

out = zeros(size(X));


for i=1:S

    aux = X(indexes{i});

    nout = sqrt(sum(aux.^2));
    normes=repmat(nout,[size(aux,1) 1]);
    I=find(normes>0);
    
    aux(I) = aux(I).*(max(0,normes(I)-lambda)./normes(I));
    
    out = out + aux(indexes_inv{i})/S;


end
