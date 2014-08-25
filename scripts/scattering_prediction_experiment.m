close all
clear all 

addpath utils
addpath stft
addpath grouplasso
addpath scatt
addpath('../video_prediction')

tmp = load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/joint.mat');

tmp.X = tmp.X ./ repmat(sqrt(sum(tmp.X.^2)),size(tmp.X,1),1) ;
tmp.Xt_same = tmp.Xt_same ./ repmat(sqrt(sum(tmp.Xt_same.^2)),size(tmp.X,1),1) ;
tmp.Xt_different = tmp.Xt_different ./ repmat(sqrt(sum(tmp.Xt_different.^2)),size(tmp.X,1),1) ;



options.M=2;
options.J=3;

psize=(2^options.J)*8;

Xbis=padarray(tmp.X,[psize 0],'replicate');

sigmas=std(Xbis,1,2);
Xbis=Xbis./repmat(sigmas,1,size(Xbis,2));

[N,L]=size(Xbis);

options.filters=morlet_filter_bank([N 1], options);
options.border = psize;

%center the filters correctly
fout=options.filters;
for r=1:size(options.filters.psi,2)
for j=1:size(options.filters.psi{r},2)
if ~isempty(options.filters.psi{r}{j})
tempo=(ifft(options.filters.psi{r}{j}{1}));
[~,reg]=max(abs(tempo));
fout.psi{r}{j}{1}=fft(circshift(tempo,-reg));
end
end
tempo=(ifft(options.filters.phi{r}));
[~,reg]=max(abs(tempo));
fout.phi{r}=fft(circshift(tempo,-reg));
end
options.filters = fout;

SX = scattbatch(Xbis, options);

[lin_pred_err, AS, Spred ] = linear_prediction(SX, [3:size(tmp.X,2)], 1);

%example of scattering prediction now.
chunk=1000;
I=randperm(size(tmp.X,2)-2);
exs = Xbis(:,I(1:chunk));
Spr = Spred(:,I(1:chunk)+1);
Sor = SX(:,I(1:chunk)+1);
oracle = Xbis(:,I(1:chunk)+1);

pred = scattbatchpredict(Spr,exs,options);




