function [out,nout]=ProximalFlat_temp(in, lambda,Gf,Gt)


aux = im2col(in,[Gf Gt],'distinct');

nout = sqrt(sum(aux.^2));
normes=repmat(nout,[size(aux,1) 1]);
I=find(normes>0);

aux(I) = aux(I).*(max(0,normes(I)-lambda)./normes(I));

out= col2im(aux,[Gf,Gt],size(in),'distinct');
