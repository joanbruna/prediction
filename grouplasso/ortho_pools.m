function Wout=ortho_pools(W,k)

[M,N]=size(W);
if k==2
%optimize this case for now
norms=sqrt(sum(W.^2,2));
Wout=W./repmat(norms,[1 N]);
W1=Wout(1:2:end,:);
W2=Wout(2:2:end,:);
dotprod=sum(W1.*W2,2);
W2 = W2 - W1.*repmat(dotprod,[1 N]);
norms=sqrt(sum(W2.^2,2));
W2=W2./repmat(norms,[1 N]);
Wout(1:2:end,:)=W1;
Wout(2:2:end,:)=W2;

else

P=M/k;
Wout=0*W;
for p=1:P
	slice{p}=W(1+(p-1)*k:p*k,:);
	[Q,~,~]=svd(slice{p}',0);
	Wout(1+(p-1)*k:p*k,:) = Q';
end

end
