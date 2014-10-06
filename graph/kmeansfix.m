function [outlabel,outm] = kmeansfix(X, k)
% Perform k-means clustering.
%   X: d x n data matrix
%   k: number of seeds
% Written by Michael Chen (sth4nth@gmail.com).
% Modified by Joan Bruna to have constant cluster sizes

n = size(X,2);
last = 0;

minener = 1e+20;
outiters=2;
maxiters=128;
rng('shuffle');

for j=1:outiters
    %s = RandStream('mt19937ar','Seed',j);
    aux=randperm(n);
    m = X(:,aux(1:k));
    %[~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1); % assign samples to the nearest centers
    %	label=label';
    [label] = constrained_assignment(X, m,n/k);
    %[label] = cheap_constrained_assignment(X, m,n/k);

    iters=0;


    while any(label ~= last) & iters < maxiters
        [u,~,label] = unique(label);   % remove empty clusters
        k = length(u);
        E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
        m = X*full(E*spdiags(1./sum(E,1)',0,k,k));    % compute m of each cluster
        last = label;
        %[~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1); % assign samples to the nearest centers
	%ener=0;label=label';
        %[label,ener] = cheap_constrained_assignment(X, m,n/k);
        [label,ener] = constrained_assignment(X, m,n/k);
	
        iters = iters +1 ;                
    end
    
    [~,~,label] = unique(label);            
    
    if ener < minener
        outlabel = label;
        outm = m;
        minener = ener;
    end            


end
fprintf('final ener is %f \n',minener)

end


%function [out,ener]=cheap_constrained_assignment(X, C, K)
%
%w=kernelizationbis(X',C');
%[N,M]=size(w); %N number of samples, M number of centers
%label=repmat([1:M],N,1);
%wmax = max(w(:))+1;
%
%[ds, I] = sort(w,2,'ascend');
%aux=I(:,1);
%ww=w;
%pp=hist(aux,[1:M]);
%
%while(max(pp)>K)
%
%J=find(pp>=K);
%for j=J
%TT=find(aux==j);
%safe=TT(1:K);
%out(safe)=j;
%
%ww(:,j)=wmax;
%ww(safe,:)=wmax;
%ww(safe,j)=0;
%
%end
%[ds, I] = sort(ww,2,'ascend');
%aux=I(:,1);
%pp=hist(aux,[1:M]);
%end
%
%ener=0;
%for n=1:N
%ener=ener+w(n,out(n));
%end
%
%end
%
%
%[~,I]=sort(w(:),'ascend');
%Ic=1+mod(I-1,N);%which sample concerns the position
%Ir=1+floor(I-1/N);%which centroid concerns the position
%
%done=0;
%rast=1;
%mask=0*I;
%ener=0;
%out=zeros(N,1);
%while done < N
%out(Ic(rast))=Ir(rast);
%ener=ener+w(I(rast));
%done=done+1;
%II=find(Ic==Ic(rast));
%mask(II)=1;
%rien=find(mask==0);
%rast=min(rien);
%end
%
%fprintf('d')
%
%end
%

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


function [ker,kern]=kernelizationbis(data,databis)

    [L,N]=size(data);
    [M,N]=size(databis);

    norms=sum(data.^2,2)*ones(1,M);
    normsbis=sum(databis.^2,2)*ones(1,L);
    ker=norms+normsbis'-2*data*(databis');
end
%aux{m}=find(out==m);
%[~,order]=sort(taille,'descend');
%clear aux;

        %w(saved,:)=maxvalue;
        %w(:,m)=maxvalue;
        %w(saved,m)=0;
        %[ds,I]=sort(w,2,'ascend');
        %out=I(:,1);

%%for m=1:M
%    aux=find(out==nextclust);
%    if length(aux)>K
%        slice=w(aux,order(m));
%        [~,tempo]=sort(slice,'ascend');
%        saved=aux(tempo(1:K));
%        out(saved)=order(m);
%        %visited(order(m))=1;
%        I(find(I==order(m)))=0;
%        for k=K+1:length(tempo)
%            i=2;
%            while I(aux(tempo(k)),i)==0
%                i=i+1;
%            end
%            out(aux(tempo(k)))=I(aux(tempo(k)),i);
%        end
%        %w(saved,:)=maxvalue;
%        %w(:,m)=maxvalue;
%        %w(saved,m)=0;
%        %[ds,I]=sort(w,2,'ascend');
%        %out=I(:,1);
%    end
%end
%
%
% 
% 
% function [outlabel,outm] = litekmeans(X, K)
% % Perform k-means clustering.
% %   X: d x n data matrix
% %   k: number of seeds
%     global norms XTX% Cashing to speed up.
%     norms = [];
%     XTX = [];
% 
%     n = size(X, 2);
%     minener = Inf;
%     outiters = 16;
%     maxiters = 1000;
%     for j = 1 : outiters
%         last = 0;
%         k = K;
%         s = RandStream('mt19937ar','Seed',j);
%         aux=randperm(s, n);
%         m = X(:,aux(1 : k));
%         label = ones(n / k, 1) * (1:k);
%         label = label(:);
%         iters = 0;
%         while any(label ~= last) && iters < maxiters
%             [u,~,label] = unique(label);   % remove empty clusters
%             k = length(u);
%             E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
%             m_piece = full(E*spdiags(1./sum(E,1)',0,k,k));
%             m = X * m_piece;    % compute m of each cluster
%             last = label;
%             [label, ener] = constrained_assignment(X, m, m_piece, n/k);
%             iters = iters + 1;
%             [~,~,label] = unique(label);
%             if ener < minener
%                 outlabel = label;
%                 outm = m;
%                 minener = ener;
%             end
%         end
%     end
% end
% 
% function [out,ener] = constrained_assignment(X, C, C_piece, K)
%     %we assign samples to the nearest centers, but with the constraint that each center receives K samples
%     w = kernelizationbis(X', C', C_piece);
%     [N,M]=size(w); %N number of samples, M number of centers
%     maxvalue = max(w(:))+1;
%     [ds,I]=sort(w,2,'ascend');    
%     out=I(:,1);
%     taille = histc(out, 1:M);
%     [hmany, nextclust]=max(taille);
%     visited=zeros(1,M);
%     go=(hmany > K);
%     choices=ones(N,1);
% 
%     while go
%         aux=find(out == nextclust);
%         slice = ds(choices(aux) * size(ds, 1) + aux) - ds((choices(aux) - 1) * size(ds, 1) + aux);
%         
%         [~, tempo]=sort(slice, 'descend');
% 
%         saved=aux(tempo(1:K));
%         out(saved)=nextclust;
% 
%         visited(nextclust)=1;
%         for k=K+1:length(tempo)         
%            vis = find(visited(I(aux(tempo(k)), 2 : end)) == 0);
%            i = vis(1) + 1;           
%            out(aux(tempo(k))) = I(aux(tempo(k)), i);
%            choices(aux(tempo(k))) = i;
%         end
%         taille = histc(out, 1:M);
%         [hmany,nextclust]=max(taille);
%         go=(hmany > K);
%     end
% 
%     ener=0;
%     for n=1:N
%         ener=ener+w(n,out(n));
%     end
% end
% 
% function ker = kernelizationbis(data, databis, databis_piece)
%     global norms XTX
%     L = size(data, 1);
%     M = size(databis, 1);
%     if (isempty(norms))
%         norms = repmat(sum(data .^ 2, 2), [1, M]);
%         XTX = data * data';
%     end
%     normsbis = repmat(sum(databis .^ 2, 2), [1, L]);
% %     ker = norms + normsbis' - 2 * data * (databis');
%     ker = norms + normsbis' - 2 * XTX * databis_piece;
% end
