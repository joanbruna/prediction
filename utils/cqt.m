function out=cqt(in, filts)

%cut in into chunks (we use 2 overlapping)
L = length(in);
T= filts.N;
J=size(filts.psi,2)+1;

nch1=floor(L/(2*T));
L1=nch1 * 2*T;

nch2=floor((L-T)/(2*T));
L2=T+ nch2*2*T;

nch = nch1+nch2;
cin=zeros(2*T, nch);

cin(:,1:2:end)=reshape(in(1:L1),2*T,nch1);
cin(:,2:2:end)=reshape(in(T+1:L2),2*T,nch2);


cf = fft(cin,[],1);
out=zeros(J,nch*T);

for j=1:J-1

tmp = ifft(cf.*repmat(filts.psi{j},1,nch));
slice=tmp(T/2+1:end-T/2,:);
out(j,:)=slice(:);

end

tmp = ifft(cf.*repmat(filts.phi,1,nch));
slice=tmp(T/2+1:end-T/2,:);
out(J,:)=slice(:);








