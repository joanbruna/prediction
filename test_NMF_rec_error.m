if ~exist('X1','var')
    % use single speaker for training
    load ../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s31.mat
    X1 = Xc;
    clear Xc;
    
    epsilon = 1;
    X1 = softNormalize(X1,epsilon);
    
    load ../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/spect_640/class_s14.mat
    X2 = Xc;

    X2 = softNormalize(X2,epsilon);
    
    
    X1 = [X1 X2];
    clear Xc X2
    
end

X = X1(:,1:end-10000);
Xv = X1(:,end-10000:end);



K = [10 50 100 500 1000];



param0 = struct;
param0.lambda = 0.1;
param0.posD = 1;
param0.pos = 1;
param0.posAlpha = 1;
param0.iter = 700;


for k=3:3
    
    param0.K = K(k);
    D = mexTrainDL(X, param0);
    
    
    A =  mexLasso(Xv,D,param0);
    
    
    n(k) = norm(Xv - D*A,'fro')/norm(Xv,'fro');
    
    keyboard
    
end



%%


%speech ='/misc/vlgscratch3/LecunGroup/bruna/grid_data/s31/pwag9a.wav';
%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s4/lrak4s.wav';
%speech ='../../../../misc/vlgscratch3/LecunGroup/bruna/grid_data/s1/lrbr4n.wav';
speech = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/s14/prin8s.wav';

params_aux = audio_config();

fs = params_aux.fs;
NFFT = params_aux.NFFT;
hop = params_aux.hop;


[x,Fs] = audioread(speech);
x = resample(x,fs,Fs);
x = x(:);


Smix = compute_spectrum(x,NFFT, hop);
Vmix = abs(Smix);
[Pmix,norms] = softNormalize(Vmix,epsilon);


A =  mexLasso(Pmix,D,param0);

a = sqrt(sum((Pmix-D*A).^2,1))./sqrt(sum((Pmix).^2,1));

norm(Pmix - D*A,'fro')/norm(Pmix,'fro')


V_rec = (D*A).*(repmat(sqrt(norms.^2+epsilon^2),size(Pmix,1),1));


y = invert_spectrum(abs(V_rec).*exp(i*angle(Smix)),NFFT , hop);




