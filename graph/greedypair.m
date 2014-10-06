function [ind, m]=greedypair(data)

[N, L0]=size(data);

L=2*floor(L0/2);
if L0 ~= L
error('input must be even size')
end

iters=10;
ener=inf;
for i=1:iters

I=randperm(L0);

seeds=I(1:L/2);
targs=I(L/2+1:L);

d1=data(:,seeds);
d2=data(:,targs);
%n1=repmat(sum(d1.^2),L/2,1);
%n2=repmat(sum(d2.^2),L/2,1);
%w=n1+n2'-2*d1'*d2;

%%assign
[out,tener]=constrained_assignment(d2,d1,1);
tind(seeds)=1:L/2;
tind(targs)=out;

tm=(d1+d2(:,out))/2;
if tener<ener
ener=tener;
ind=tind;
m=tm;
end
end
fprintf('fnale ener is %f \n', ener)

end


function [out,ener]=constrained_assignment(X, C, K)
%we assign samples to the nearest centers, but with the constraint that each center receives K samples
w=kernelizationbis(X',C');
[N,M]=size(w); %N number of samples, M number of centers

maxvalue = max(w(:))+1;

[ds,I]=sort(w,2,'ascend');
%[ds2,I2]=sort(w,1,'ascend');

out=I(:,1);
for m=1:M
    taille(m)=length(find(out==m));
end
[hmany,nextclust]=max(taille);

visited=zeros(1,M);


go=(hmany > K);
choices=ones(N,1);

while go
    %fprintf('%d %d \n', nextclust, hmany)
    aux=find(out==nextclust);

    for l=1:length(aux)
        slice(l) = ds(aux(l),choices(aux(l))+1)-ds(aux(l),choices(aux(l)));
    end
    [~,tempo]=sort(slice,'descend');
    clear slice;
    %slice=w(aux,nextclust);
    %[~,tempo]=sort(slice,'ascend');
    
    saved=aux(tempo(1:K));
    out(saved)=nextclust;

    visited(nextclust)=1;
    for k=K+1:length(tempo)
       i=2;
       while visited(I(aux(tempo(k)),i)) 
          i=i+1;
       end
       out(aux(tempo(k)))=I(aux(tempo(k)),i);
       choices(aux(tempo(k)))=i;
    end
    for m=1:M
        taille(m)=length(find(out==m));
    end
    [hmany,nextclust]=max(taille);
    go=(hmany > K);
end

ener=0;
for n=1:N
ener=ener+w(n,out(n));
end

end


