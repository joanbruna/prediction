function [ker,kern]=kernelization(data)

[L,N]=size(data);

norms=sum(data.^2,2)*ones(1,L);
ker=norms+norms'-2*data*(data');

kern=sqrt(ker./(norms+norms'));


