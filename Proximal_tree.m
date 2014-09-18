function out = Proximal_tree(X,indexes,indexes_inv, lambda)


S = length(indexes);
J = size(indexes{1},2);

out = zeros(size(X));


for i=1:S

    tmp = X;
    for j=1:J

    aux = tmp(indexes{i}{j});

    nout = sqrt(sum(aux.^2));
    normes=repmat(nout,[size(aux,1) 1]);
    I=find(normes>0);
    
    aux(I) = aux(I).*(max(0,normes(I)-lambda)./normes(I));
    
    tmp=aux(indexes_inv{i}{j});
    end
    out = out + tmp/S;

end
