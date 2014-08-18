
function output = testFile(speech,noise,testFun,D,nparam,SNR_dB)


params_aux = audio_config();

fs = params_aux.fs;
NFFT = params_aux.NFFT;
hop = params_aux.hop;


[x,Fs] = audioread(speech);
x = resample(x,fs,Fs);
x = x(:);


[n,Fs] = audioread(noise);
n = resample(n,fs,Fs);
n = n(:);



% adjust the size
m = min(length(x),length(n));

x = x(1:m);
n = n(1:m);

x = x - mean(x);
n = n - mean(n);

% adjust SNR
x = x/sqrt(sum(power(x,2)));
if sum(power(n,2))>0
    n = n/sqrt( sum(power(n,2)));
    n = n*power(10,(-SNR_dB)/20);
end

% compute noisy signal
mix = x + n;

Smix = compute_spectrum(mix,NFFT, hop);
Vmix = abs(Smix);

Sx = compute_spectrum(x,NFFT, hop);
Vx = abs(Sx);
Px = mexNormalize(Vx);

Sn = compute_spectrum(n,NFFT, hop);
Vn = abs(Sn);
Pn = mexNormalize(Vn);


%

rep = 1;
rates = zeros(rep,3);
obj = zeros(rep,1);
rec = obj;


add_frame = mod(size(Vmix,2),2);

if add_frame
    Vmix(:,end+1) = 0;
end
    

for i=1:rep


Pmix = mexNormalize(Vmix);

[Hs,Hn,Wn,obj_i] = testFun(Pmix,nparam,Px,Pn);


if add_frame
    Hs = Hs(:,1:end-1);
    Hn = Hn(:,1:end-1);
end


R = {};
R{1} = D* Hs;
R{2} = Wn* Hn;

y_out = wienerFilter2(R,Smix);


m = length(y_out{1});
x2 = x(1:m);
n2 = n(1:m);

[SDR,SIR,SAR,perm] = bss_eval_sources( [y_out{1},y_out{2}]',[x2,n2]');


rates(i,:) = [SDR(1) SIR(1) SAR(1)];

obj(i) = obj_i;

rec(i) = norm(Px - R{1},'fro');

end

output.A = Hs;
output.obj = obj;
output.Wout = Wn;
output.speech = y_out{1};
output.speech_orig = x2;
output.mix = mix;
output.rates = rates;

end
