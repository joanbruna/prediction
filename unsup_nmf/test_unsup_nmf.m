

file = 'unsup_nmf/sunrise22k100-mix.wav';

params_aux = audio_config();

fs = params_aux.fs;
NFFT = params_aux.NFFT;
hop = params_aux.hop;


[x,Fs] = audioread( file );
x = resample(x,fs,Fs);
tseg = 10;
x = x(1:tseg*fs);
x = x(:);


Sx = compute_spectrum(x,NFFT, hop);
Vx = abs(Sx);

epsilon = 1;
[X,n] = softNormalize(Vx,epsilon);


% --------------------


options.K=20;
options.epochs=0.5;
options.nmf = 1;
options.alpha_iters=80;
options.batchsize=size(X,2);
options.sort_dict = 1;
options.plot_dict = 0;
options.lambda = 0.1;
options.mu = 0.5;


% Train initial dictionary only with slowness and NMF initialization
options.init_nmf = 1;
options.use_flow = 1;


[D,Dnmf] = train_nmf_optflow(X, options);


% Train Dictionary with flow using slow dictionary as input
%%



ptheta = struct;
ptheta.sigma = 1;
ptheta.hn = 11;
ptheta.lambda = 0.1;
ptheta.lambdar = 0.00001;

options.lambda_t = ptheta.lambda;
options.lambda_tr = ptheta.lambdar;
options.hn = ptheta.hn;
options.sigma = ptheta.sigma;


alpha = zeros(size(D,2),size(X,2));
theta = alpha;
for j = 1:3
    
    [alpha,cost_aux,Salpha] = nmf_optflow( X, D, theta, options);
    
    if options.use_flow
        %theta = optflow_taylor2(alpha, ptheta,theta);
        theta = optflow_taylor_temp(alpha, ptheta);
    end
    
end


%%


R = {};
R{1} = D(:,4)*alpha(4,:);
R{2} = D(:,13:17)*alpha(13:17,:);

y_out = wienerFilter2(R,Sx);



