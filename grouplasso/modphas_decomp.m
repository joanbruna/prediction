function [modul,phas,reco]=modphas_decomp(alpha, groupsize)

[K,M]=size(alpha);

if groupsize==2

X = alpha(1:2:end,:) + i*alpha(2:2:end,:);
modul=abs(X);
phas=angle(X);

elseif groupsize>2

tmp=reshape(alpha,groupsize,numel(alpha)/groupsize);
modul=reshape(sqrt(sum(tmp.^2)),K/groupsize,M);

if mod(K,groupsize)>0
error('something is fishy')
end

reco=0*alpha;
for g=1:groupsize
reco(g:groupsize:end,:)=modul;
end

I=find(reco>0);
phas=0*alpha;
phas(I)=alpha(I)./reco(I);


else

modul=abs(alpha);
phas=sign(alpha);
reco=modul;

end




