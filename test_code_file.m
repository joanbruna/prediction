
% Speaker Not used at training
%speech ='/misc/vlgscratch3/LecunGroup/bruna/grid_data/s4/lrak4s.wav';
speech ='/misc/vlgscratch3/LecunGroup/bruna/grid_data/s1/lrbr4n.wav';

% Speaker Used at training
%speech ='/misc/vlgscratch3/LecunGroup/bruna/grid_data/s31/pwag9a.wav';
%speech = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/s14/prin8s.wav';
%speech = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/s14/bbbk9p.wav';

% Noise signal
%speech = '/misc/vlgscratch3/LecunGroup/bruna/noise_data/train/noise_sample_08.wav';
%speech = '/misc/vlgscratch3/LecunGroup/bruna/noise_data/babble/noise_sample_08.wav';


params_aux = audio_config();

fs = params_aux.fs;
NFFT = params_aux.NFFT;
hop = params_aux.hop;


[x,Fs] = audioread(speech);
x = resample(x,fs,Fs);
x = x(:);


Sx = compute_spectrum(x,NFFT, hop);
Vx = abs(Sx);

epsilon = 1;
[X,n] = softNormalize(Vx,epsilon);



%% Coding


options.K=100;
options.epochs=2;
options.nmf = 1;
options.alpha_iters=80;
options.batchsize=256;
options.sort_dict = 1;
options.plot_dict = 0;
options.lambda = 0.1;
options.mu = 0.5;


ptheta = struct;
ptheta.sigma = 1;
ptheta.hn = 11;
ptheta.lambda = 0.1;
ptheta.lambdar = 0.00001;

% options.lambda_t = ptheta.lambda;
% options.lambda_tr = ptheta.lambdar;
% options.hn = ptheta.hn;
% options.sigma = ptheta.sigma;


[A,theta] = nmf_optflow_smooth(X,D,options,ptheta);
