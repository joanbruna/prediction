function out=compute_grad_tau(N)
%this function computes the linear operator regularizing the optical flow

chunk=zeros(N);
chunk2=zeros(N);
aux=zeros(N^2);
auxbis=zeros(N^2);
rast=1;
for i=1:N
        chunk(i,j)=1;
        chunk2(i,j)=1;
        if i>1
        chunk(i-1,j)=-1;
        end
        aux(rast,:)=chunk(:);
        if j>1
        chunk2(i,j-1)=-1;
        end
        auxbis(rast,:)=chunk2(:);
        rast=rast+1;
        chunk=0*chunk;
        chunk2=0*chunk2;
     end
end

tempo = [aux ; auxbis];

out=zeros(2*size(tempo));

out(1:size(tempo,1),1:size(tempo,2))=tempo;
out(size(tempo,1)+1:end,size(tempo,2)+1:end)=tempo;

