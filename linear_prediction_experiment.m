close all
clear all 

addpath utils
addpath stft
addpath grouplasso
addpath('../video_prediction')

tmp = load('/misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/joint.mat');

tmp.X = tmp.X ./ repmat(sqrt(sum(tmp.X.^2)),size(tmp.X,1),1) ;
tmp.Xt_same = tmp.Xt_same ./ repmat(sqrt(sum(tmp.Xt_same.^2)),size(tmp.X,1),1) ;
tmp.Xt_different = tmp.Xt_different ./ repmat(sqrt(sum(tmp.Xt_different.^2)),size(tmp.X,1),1) ;

[lin_pred_err, A ] = linear_prediction(tmp.X, [3:size(tmp.X,2)], 1);

%evaluate it in both test sets
L=size(tmp.Xt_same,2);
rien=[tmp.Xt_same(:,2:L-1) ; tmp.Xt_same(:,1:L-2)];
Pred = A * rien;
Ref = tmp.Xt_same(:,3:L);
erro=norm(Ref(:)-Pred(:))/norm(Ref(:));
rien = tmp.Xt_same(:,2:L-1);
naive = norm(Ref(:)-rien(:))/norm(Ref(:));
fprintf('linear prediction (2 context frames) error is %f past is %f \n', erro, naive);




L=size(tmp.Xt_different,2);
rien=[tmp.Xt_different(:,2:L-1) ; tmp.Xt_different(:,1:L-2)];
Pred = A * rien;
Ref = tmp.Xt_different(:,3:L);
erro=norm(Ref(:)-Pred(:))/norm(Ref(:));
rien = tmp.Xt_different(:,2:L-1);
naive = norm(Ref(:)-rien(:))/norm(Ref(:));
fprintf('Different speaker: linear prediction (2 context frames) error is %f past is %f \n', erro, naive);











