
Z = Zout;
f = 0;

Poole = sqrt(conv2(Z.^2+eps,box,'valid'));
for k=1:4
    
    
    tPoole=Poole(off1(k):groupsize:end,off2(k):time_groupsize:end);
    
    fp = zeros(size(tPoole));
    id = find(tPoole>0);
    fp(id) = 1./tPoole(id);
    
    aux = 0*aux;
    aux(off1(k):groupsize:end,off2(k):time_groupsize:end)=fp;
    uPoole = (conv2(aux,box,'full')).*Z;
    
    f = f + sum(uPoole(:));
    
end