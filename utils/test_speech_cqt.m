


root = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/';

fs = 16000;
NFFT = 1024;
hop = NFFT/4;


folder1 = sprintf('%ss2/',root);
d1 = dir(sprintf('%s%s',folder1,'*.wav'));

[x1,Fs] = audioread(sprintf('%s%s',folder1,d1(999).name));
x1 = resample(x1,fs,Fs); T1 = length(x1);


folder2 = sprintf('%ss4/',root);
d2 = dir(sprintf('%s%s',folder2,'*.wav'));

[x2,Fs] = audioread(sprintf('%s%s',folder2,d2(999).name));
x2 = resample(x2,fs,Fs); T2 = length(x2);

T = min(T1,T2);

x1 = x1(1:T);
x2 = x2(1:T);

mix = x1+x2;

X = compute_spectrum(mix,NFFT,hop);
X1 = compute_spectrum(x1,NFFT,hop);
X2 = compute_spectrum(x2,NFFT,hop);


%%CQT-joan
options.null=0;
filts=cqt_prepare(options);

Qj = cqt(mix,filts);
Q1j = cqt(x1,filts);
Q2j = cqt(x2,filts);


%
%
%% CQT



% fmin = 40;
% fmax = 8000;
% pstep = 1/48;       % 0.25 semitone
% np = 50;

fmin = 40;
fmax = fs/2;
pstep = 1/32;       % 0.25 semitone
np = 15;
                        % Periods per window

lmin = 0.02;                     % min window length
lmax = .25;                       % max window length

[W,fw,tw] = constQmtx(fs,fmin,fmax,pstep,np,lmin,lmax);




Nwin  = size(W,2);                % Window size
tstep = 256/fs;                     % Time step
hop = round(tstep*fs);          % Samples
tstep = hop/fs;

[Q,t] = constQ(mix, W, hop);

[Q1,t] = constQ(x1, W, hop);

[Q2,t] = constQ(x2, W, hop);

%figure(1)
%subplot(311)
%imagesc(Q)
%subplot(312)
%imagesc(Q1)
%subplot(313)
%imagesc(Q2)
%
%figure(2)
%subplot(311)
%imagesc(abs(X))
%subplot(312)
%imagesc(abs(X1))
%subplot(313)
%imagesc(abs(X2))
%
%% figure(1)
%% subplot(311)
%% imagesc(log(0.1+abs(Q)))
%% subplot(312)
%% imagesc(log(0.1+abs(Q1)))
%% subplot(313)
%% imagesc(log(0.1+abs(Q2)))
%% 
%% figure(2)
%% subplot(311)
%% imagesc(log(0.1+abs(X)))
%% subplot(312)
%% imagesc(log(0.1+abs(X1)))
%% subplot(313)
%% imagesc(log(0.1+abs(X2)))
%
%figure(1); figure(2)
