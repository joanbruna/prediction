
function  result = denoising_test(testFun,test_files,noise,options)

% SNR of the mixture
SNR_dB = getoptions(options,'SNR_dB',1);

% use spectrum or scatt
is_stft = getoptions(options,'is_stft',1);

% display results
verbose  = getoptions(options,'verbose',1);

% which audio files to use
idf = getoptions(options,'idf',[1,1,1]);

% representation parameters
if is_stft
    NFFT = getoptions(options,'NFFT',1024);
    fs = getoptions(options,'fs',1600);
    hop = getoptions(options,'hop',NFFT/2);
    epsilon = getoptions(options,'epsilon',1e-3);

else
    Npad = getoptions(options,'Npad',2^17);
    scparam = getoptions(options,'scparam',[]);
    scparam.N = Npad;
    filts = cqt_prepare(scparam);
    epsilon = getoptions(options,'epsilon',1e-3);
    
    pp.scparam = scparam;
    pp.filts = filts;
end


M = length(noise);
N = size(idf,1);
P = length(SNR_dB);

output = cell(M,N,P);

SDR = zeros(M,N,P);
NSDR = zeros(M,N,P);
SIR = zeros(M,N,P);
SAR = zeros(M,N,P);

for j=1:M
for i=1:N

    % get files
    test_file_name = test_files{idf(i,1)}{idf(i,2)}{idf(i,3)}.speaker;
    x1 = test_files{idf(i,1)}{idf(i,2)}{idf(i,3)}.x;
    x2 = noise{j}.x;
    
    T = min(length(x1),length(x2));
    x1 = x1(1:T)/norm(x1);
    x2 = x2(1:T);
    
    if verbose
       fprintf('------------------------------\n')
       fprintf('File: %s, noise %s\n\n',test_file_name,noise{j}.file)
    end
    
    % compute separation for several SNRs
    for k=1:P
    
    x2_ = x2/norm(x2)*power(10,(-SNR_dB(k))/20);
    mix = x1 + x2_;

    %mix = (x1/norm(x1)+x2/norm(x2));
    if is_stft
        X = compute_spectrum(mix,NFFT,hop);
        Xn = softNormalize(abs(X),epsilon);
        
    else
        [X, phmix] = batchscatt(pad_mirror(mix',Npad),filts, scparam);
        Xn = softNormalize(abs(X),epsilon);
    end
    
    
    %[Hs,Hn,Wn,obj] = denoising_nmf(Xn,D,nparam);
    [W1H1,W2H2] = testFun(Xn);
    
    % wiener filter
    eps_1 = 1e-6;
    V_ap = W1H1.^2 +W2H2.^2 + eps_1;
    
    SPEECH1 = ((W1H1.^2)./V_ap).*X;
    SPEECH2 = ((W2H2.^2)./V_ap).*X;
    
    if is_stft
        speech1 = invert_spectrum(SPEECH1,NFFT,hop,T);
        speech2 = invert_spectrum(SPEECH2,NFFT,hop,T);
        
    else
        
        speech1 = audioreconstruct(SPEECH1, pp, phmix);
        speech2 = audioreconstruct(SPEECH2, pp, phmix);
        
        speech1 = speech1(1:T)';
        speech2 = speech2(1:T)';
        
    end

    Parms =  BSS_EVAL(x1', x2_', speech1', speech2', mix');
    
    Parms.SDR = Parms.SDR(2);
    Parms.SIR = Parms.SIR(2);
    Parms.SAR = Parms.SAR(2);
    Parms.NSDR = Parms.NSDR(2);
    Parms.SNR_in = snr(x1,x2_);
    Parms.SNR_out = snr(x1,x1-speech1);
    
    if verbose
       disp(Parms) 
    end
    
    
    SDR(j,i,k) = Parms.SDR;
    NSDR(j,i,k) = Parms.NSDR;
    SAR(j,i,k) = Parms.SAR;
    SIR(j,i,k) = Parms.SIR;

    output{j,i,k} = Parms;
    
    end
end
end

result.SDR = SDR;
result.NSDR = NSDR;
result.SIR = SIR;
result.SAR = SAR;
result.output = output;

% globl variables
stat = cell(1,P);
for k=1:P
    aux = SDR(:,:,k);
    stat{k}.mean_SDR = mean(aux(:));
    stat{k}.std_SDR = std(aux(:));
    
    aux = SIR(:,:,k);
    stat{k}.mean_SIR = mean(aux(:));
    stat{k}.std_SIR = std(aux(:));
    
    aux = SAR(:,:,k);
    stat{k}.mean_SAR = mean(aux(:));
    stat{k}.std_SAR = std(aux(:));
    
    aux = NSDR(:,:,k);
    stat{k}.mean_NSDR = mean(aux(:));
    stat{k}.std_NSDR = std(aux(:));
        
end

result.stat = stat;



