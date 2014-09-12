function [V, S, L, spect] = graphlaplacian(datacov, options)

if 0
matdir='~/matlab';
vl_setup
%tree=vl_kdtreebuild(data);
j1 = getoptions(options,'num_neighbors',32);
%[nnid, ndist] = vl_kdtreequery(tree,data,data, 'NUMNEIGHBORS',j1,'MAXCOMPARISONS',600) ;
[~, idtmp]=sort(datacov,'ascend');
nnid=idtmp(:,1:j1);

opts.kNN=j1;
opts.alpha=1;
opts.kNNdelta=j1;
S=fgf(data',opts,nnid');
else

%construct a weight matrix from datacov
%we find sigma as a percentile of each row
[sval, idtmp]=sort(datacov,2,'ascend');
j1 = getoptions(options,'num_neighbors',12);
sigmas=sval(:,j1);
S=zeros(size(datacov));
for t=1:size(datacov,2)
S(t,:)=exp(-datacov(t,:)/sigmas(t));
end
S=(S+S')/2;
end

D = diag(sum(S).^(-1/2));
L = eye(size(S,1)) - D * S * D;
[V,ev]=eig(L);
spect = diag(ev);

ncomp = getoptions(options,'nspectcomp',round(size(datacov,2)/2));
V=V(:,1:ncomp);


%w=kernelization(zpool);
%[valos,aux]=sort(w,2,'descend');
%
%MM=128;
%NN=128;
%MAXiter = 1000; % Maximum iteration for KMeans Algorithm
%REPlic = 10; % Replication for KMeans Algorithm
%pp=100;
%sigma = (mean(valos(:,pp)));
%S = exp(-w.^2/(2*sigma^2));
%DD = diag(sum(S));
%DDbis = diag(sum(S).^(-1/2));
%Lbis = eye(size(S,1)) - DDbis * S * DDbis;
%L = DD - S;
%[ee,ev]=eig(L);
%W0=ee(:,1:MM);
%
%[idx,W1] = kmeans(W0,NN,'start','sample','maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
%W1=W1';
%
%clval=0.01;
%figure;scatter3(max(-clval,min(clval,W0(:,2)-median(W0(:,2)))),max(-clval,min(clval,W0(:,3)-median(W0(:,3)))),max(-clval,min(clval,W0(:,4)-median(W0(:,4)))))
%
%

