
function output = testFile(speech,noise,fun,params,SNR_dB)

if varargin == 4
    SNR = 0;
end

[x,Fs] = audioread(speech);
if Fs~= params.fs
    x = resample(x,Fs,params.fs);
    x = x(:);
end

if ~isempty(noise)
    [n,Fs] = audioread(speech);
    if Fs~= params.fs
        n = resample(n,Fs,params.fs);
        n = n(:);
    end
        
else
    
end


% adjust the size
m = min(length(x),length(n));

x = x(1:m);
n = n(1:m);

% adjust SNR
x = x/sqrt(sum(power(x,2)));
if sum(power(n,2))>0
    n = n/sqrt( sum(power(n,2)));
    n = n*power(10,(-SNR_dB)/20);
end

% compute noisy signal
mix = x+ n;


% compute spectral representation
Smix = params.scf * stft(x, params.NFFT , params.winsize, params.hop);

% Compute unmixing
[Ps,Pn] = fun(Smix, params.fun_params);


n1 = size(dic_v.Wv , 2);

Vv = dic_v.Wv* Hmix(1:n1,:);
Vn = dic_n.Wn* Hmix((n1+1):end,:);


% initial SNR, SDR, etc
sdr_i = bss_eval_sources(mix',yv');
snr_i = get_SNR(mix',yv');


[rates, signals] = evaluate_unmixing(Vv, Vn, yv, yn, Smix, speaker.NFFT, speaker.step);

orig{1} = x;
orig{2} = n;


output.rates = rates;

output.sdr_i = sdr_i;
output.snr_i = snr_i;

output.mix = mix;
output.orig = orig;
output.signals = signals;
output.type = type;
