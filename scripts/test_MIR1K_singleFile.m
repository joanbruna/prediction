

params_aux = audio_config();

fs = params_aux.fs;
NFFT = params_aux.NFFT;
hop = params_aux.hop;

epsilon = 10;


FilePath='/misc/vlgscratch3/LecunGroup/pablo/MIR1K/Wavfile/';

files=dir([FilePath,'*.wav']);

epslion =10;

count = 1;
pt = 1;

i = 811;
disp(test_files(i).name)

%[x,Fs] = audioread([FilePath,files(i).name]);
[x,Fs] = audioread([FilePath,test_files(i).name]);

wavinA= x(:,1);
wavinE= x(:,2);
mix=wavinA+wavinE;


% mix
mix = resample(mix,fs,Fs);

S = compute_spectrum(mix,NFFT, hop);
X = abs(S);

X = softNormalize(X,epsilon);

%============

% options.W = D_back;
% options.tau = 0.5;
%options_optflow.mu = 1;

[A,theta,SA,An] = nmf_optflow_smooth(X,Dslow,options_optflow,ptheta);


R = {};
R{1} = Dnmf* A;
R{2} = D_back*An;

y_out = wienerFilter2(R,S);
y_out_of = y_out;

m = length(y_out{1});
x2 = wavinA(1:m);
n2 = wavinE(1:m);


[SDR_,SIR_,SAR_,perm] = bss_eval_sources( mix', wavinE' );


[SDR,SIR,SAR,perm] = bss_eval_sources( [y_out{1},y_out{2}]',[x2,n2]');

disp('NMF:')
[SDR,SIR,SAR]


if pt
    xv = resample(wavinE,fs,Fs);
    Sv = compute_spectrum(xv,NFFT, hop);
    Xv = abs(Sv);
    Xv = softNormalize(Xv,epsilon);
    
    figure(1)
    subplot(311)
    dbimagesc(X+0.001);
    subplot(312)
    dbimagesc(Xv+0.001);
    subplot(313)
    dbimagesc(Dslow* A+0.001);
    
    figure(2)
    subplot(411)
    dbimagesc(X+0.001);
    subplot(412)
    imagesc(A);
    subplot(413)
    imagesc(SA);
    
    
end


%============


[A,theta,SA,An] = nmf_optflow_smooth(X,Dnmf,options_nmf,ptheta);


R = {};
R{1} = Dnmf* A;
R{2} = D_back*An;

y_out = wienerFilter2(R,S);
y_out_nmf = y_out;

m = length(y_out{1});
x2 = wavinA(1:m);
n2 = wavinE(1:m);

[SDR_nmf,SIR_nmf,SAR_nmf,perm] = bss_eval_sources( [y_out{1},y_out{2}]',[x2,n2]');

disp('NMF:')
[SDR_nmf,SIR_nmf,SAR_nmf]

%==========


if pt
    
    
    subplot(414)
    imagesc(A);
    
    figure(3)
    subplot(311)
    dbimagesc(X+0.001);
    subplot(312)
    dbimagesc(Xv+0.001);
    subplot(313)
    dbimagesc(Dslow* A+0.001);
end

%fprintf('NSDR voice: %f\nNSDR music: %f\n',NSDR(1),NSDR(2));

