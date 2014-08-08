function [X,templates,phaschange,dewhiten,aviam]=genetate_jitter_data(options)

N=getoptions(options,'N',128);
L=getoptions(options,'L',2^21);
K=getoptions(options,'Ksmooth',8);

ntemplates=getoptions(options,'ntemplates',4);

template_smoothness = getoptions(options,'spatial_smoothness',8);
marge = N/ntemplates;

maxjitter = marge/2;
maxjitter0 = marge/4;

templates=randn(ntemplates,marge/2);
h=ones(1,template_smoothness);
%h=gausswin(maxjitter0, template_smoothness);
h=h/sum(h);
templates = conv2(templates, h, 'same');

jitters = randn(ntemplates, L);
htime = K^(-1)*ones(1,K);
jitters = conv2(randn(ntemplates,L),htime,'same');
maxval = max(abs(jitters(:)));
jitters = (maxjitter0/maxval) * jitters;

X=zeros(N,L);

xt=randn(maxjitter,1);
x0=zeros(marge,1);
x0(maxjitter/2+1:end-maxjitter/2)=xt;
x1=zeros(marge,1);
x1(maxjitter/2+0:end-maxjitter/2-1)=xt;
xf0=fft(x0);
xf1=fft(x1);
phaschange=xf1./xf0;

PP=repmat(phaschange,1,L);

jitters=1*jitters - maxjitter0;
for n=1:ntemplates
cosa=fft(templates(n,:),marge);

aviam= repmat(transpose(cosa),1,L) .* (PP .^ repmat(jitters(n,:),marge,1));
X(1+(n-1)*marge:n*marge,:)= ifft(aviam,[],1);

end
X=real(X);


%%%whiten data
if 0
size(X)
[U, S, V] = svd( X', 0);
X = U(:,1:kk);
dewhiten = V(:,1:kk)*S(1:kk,1:kk);
%X = U * S * V';

end






