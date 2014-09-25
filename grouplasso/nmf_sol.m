function out = nmf_sol(Z2,pZ,dfp,Dgn,lambda2gn,groupsize,time_groupsize)

N = size(Dgn,1);
M = groupsize*N/2;

R = size(Z2,2);

if size(Z2,2)>1
    out = zeros(N,R);
    for i=1:size(Z2,2)
         out(:,i) = nmf_grad(Z2(:,i),pZ(:,i),dfp(:,i),Dgn,lambda2gn,groupsize,time_groupsize);
    end
    return
end


id = find(Z2>0);
if ~isempty(id)
Did = Dgn(:,id);

A = Did*((Did'*Did + lambda2gn*eye(length(id)))\Did');

pZi = zeros(size(pZ));
ii = (pZ~=0);
pZi(ii) = 1./pZ(ii);

B = A.*repmat(pZi',N,1);
out = B'*(pZ.*dfp);


else
    out = zeros(N,1);
end

end