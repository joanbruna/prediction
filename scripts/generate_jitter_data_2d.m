function [X,templates,phaschange,dewhiten,aviam]=generate_jitter_data2d(options)


N=getoptions(options,'N',16);
L=getoptions(options,'L',2^17);
K=getoptions(options,'Ksmooth',32);

ntemplates=getoptions(options,'ntemplates',1);

template_smoothness = getoptions(options,'spatial_smoothness',3);
marge = N;
tempsize = 2;
h=ones(template_smoothness);
h=h/(template_smoothness*template_smoothness);

maxjitter = marge/2;
maxjitter0 = marge;

X=zeros(N*N,L);

%xt=randn(maxjitter);
xt=zeros(maxjitter);
xt(maxjitter/2,maxjitter/2)=1;
x0=zeros(marge);
x0(marge/2-maxjitter/2+1:marge/2+maxjitter/2,marge/2-maxjitter/2+1:marge/2+maxjitter/2)=xt;
x1=zeros(marge);
x1(marge/2-maxjitter/2+0:marge/2+maxjitter/2-1,marge/2-maxjitter/2+1:marge/2+maxjitter/2)=xt;
x2=zeros(marge);
x2(marge/2-maxjitter/2+1:marge/2+maxjitter/2,marge/2-maxjitter/2+0:marge/2+maxjitter/2-1)=xt;

xf0=fft2(x0);
xf1=fft2(x1);
xf2=fft2(x2);
phaschange1=xf1./xf0;
phaschange2=xf2./xf0;

PP1=repmat(phaschange1(:),1,L);
PP2=repmat(phaschange2(:),1,L);

for n=1:ntemplates

template=abs(randn(tempsize));
template = conv2(template, h, 'full');

htime = K^(-1)*ones(1,K);
jitters = conv2(randn(2,L),htime,'same');
maxval = max(abs(jitters(:)));
jitters = (maxjitter0/maxval) * jitters;

%try just a 1d jitter model
%jitters(2,:)=rand*jitters(1,:);


cosa = fft2(template,marge,marge);

aviam = repmat(cosa(:),1,L) .* (PP1.^repmat(jitters(1,:),marge*marge,1)) .* (PP2.^repmat(jitters(2,:),marge*marge,1)) ;

%ifft2
avi = reshape(aviam,N,numel(aviam)/N);
avi = ifft(avi,[],1);
avi = reshape(avi,N, N, L);
avi = permute(avi,[2 1 3]);
avi = ifft(avi, [], 1);
avi = permute(avi, [2 1 3]);

X = X + abs(reshape(avi,N*N,L));

end
%X=abs(X);



%%%whiten data
if 0
size(X)
[U, S, V] = svd( X', 0);
X = U(:,1:kk);
dewhiten = V(:,1:kk)*S(1:kk,1:kk);
%X = U * S * V';

end






