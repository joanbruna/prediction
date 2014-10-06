function [W,f,t] = constQmtx(fs,fmin,fmax,pstep,np,lmin,lmax)

pmin = ceil(log2(fmin/440));
pmax = floor(log2(fmax/440));
f = 2.^[pmin:pstep:pmax]*440;

len = fs*np./f;                   % Window length
len = min(lmax*fs, max(lmin*fs, len));
len = floor(len/2)*2+1;

t = [0:max(len)-1]-max(len)*0.5;
T = bsxfun(@rdivide, t, len(:));
W = (0.54+0.46*cos(2*pi*T)).*(abs(T) <= 0.5);
W = bsxfun(@rdivide, W, sum(W,2));
W = W.*exp(-2*pi*sqrt(-1)*f(:)*t(:)'/fs);

t = t/fs;