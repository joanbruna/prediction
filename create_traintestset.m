close all
clear all

addpath utils
addpath stft

numspeakers=20;
numclips=300;
numtclips=100;
numtdclips=200;

totspeakers=34;

I0=randperm(totspeakers);
I=I0(1:numspeakers);
Id = I0(numspeakers+1:end);

%create train and test
X=zeros(321,800000);
Xt_same=zeros(321,200000);
Xt_different=0*Xt_same;

rast=1;
rastt=1;
for i=I
tmp = load(sprintf('/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s%d.mat',i));
totclips=max(tmp.met);
II=randperm(totclips);
for ii=1:min(numclips,totclips)
JJ = find(tmp.met==II(ii));
X(:,rast:rast+length(JJ)-1) = tmp.Xc(:,JJ);
rast=rast+length(JJ);
end
st=ii;
for ii=st+1:min(st+numtclips,totclips)
JJ = find(tmp.met==II(ii));
Xt_same(:,rastt:rastt+length(JJ)-1) = tmp.Xc(:,JJ);
rastt=rastt+length(JJ);
end
i
end

X = X(:,1:rast-1);
Xt_same = Xt_same(:,1:rastt-1);

rast=1;
for i=Id
tmp = load(sprintf('/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s%d.mat',i));
totclips=max(tmp.met);
II=randperm(totclips);
for ii=1:min(numtdclips,totclips)
JJ = find(tmp.met==II(ii));
Xt_different(:,rast:rast+length(JJ)-1) = tmp.Xc(:,JJ);
rast=rast+length(JJ);
end
i
end

Xt_different = Xt_different(:,1:rast-1);




