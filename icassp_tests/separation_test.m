
function  result = separation_test(testFun,test_files_1,test_files_2,options)

% SNR of the mixture - Always fixed
SNR_dB = getoptions(options,'SNR_dB',0);


Npad = getoptions(options,'Npad',2^16);

% use spectrum or scatt
is_stft = getoptions(options,'is_stft',1);

% display results
verbose  = getoptions(options,'verbose',1);

% which audio files to use
id1 = getoptions(options,'id1',[1,1,1]);
id2 = getoptions(options,'id2',[1,1,1]);

% representation parameters
if is_stft
    
    NFFT = getoptions(options,'NFFT',1024);
    fs = getoptions(options,'fs',1600);
    hop = getoptions(options,'hop',NFFT/2);
    epsilon = getoptions(options,'epsilon',1e-3);
    
end


M = size(id2,1);
N = size(id1,1);


output = cell(M,N);

SDR = zeros(M,N);
NSDR = zeros(M,N);
SIR = zeros(M,N);
SAR = zeros(M,N);


for j=1:M
for i=1:N

    % get files
    test_file_name_1 = test_files_1{id1(i,1)}{id1(i,2)}{id1(i,3)}.speaker;
    x1 = test_files_1{id1(i,1)}{id1(i,2)}{id1(i,3)}.x;
    
    test_file_name_2 = test_files_2{id2(j,1)}{id2(j,2)}{id2(j,3)}.speaker;
    x2 = test_files_2{id2(j,1)}{id2(j,2)}{id2(j,3)}.x;
    
    T = min([length(x1),length(x2),Npad]);
    x1 = x1(1:T)/norm(x1(1:T));
    x2 = x2(1:T);
    
    if verbose
       fprintf('------------------------------\n')
       fprintf('File: %s, File %s\n\n',test_file_name_1,test_file_name_2)
    end
    
    
    x2 = x2/norm(x2)*power(10,(-SNR_dB)/20);
    mix = x1 + x2;

    %mix = (x1/norm(x1)+x2/norm(x2));
    if is_stft
        X = compute_spectrum(mix,NFFT,hop);
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
    end

    Parms =  BSS_EVAL(x1', x2', speech1', speech2', mix');
    
    Parms.SNR_in = snr(x1,x2);
    Parms.SNR_out = [snr(x1,x1-speech1) snr(x2,x2-speech2)];
    
    if verbose
       disp(Parms) 
    end
    
    
    SDR(j,i) = mean(Parms.SDR);
    NSDR(j,i) = mean(Parms.NSDR);
    SAR(j,i) = mean(Parms.SAR);
    SIR(j,i) = mean(Parms.SIR);

    output{j,i} = Parms;
    
    
end
end


result.SDR = SDR;
result.NSDR = NSDR;
result.SIR = SIR;
result.SAR = SAR;
result.output = output;

% globl variables

stat.mean_SDR = mean(SDR(:));
stat.std_SDR = std(SDR(:));

stat.mean_SIR = mean(SIR(:));
stat.std_SIR = std(SIR(:));

stat.mean_SAR = mean(SAR(:));
stat.std_SAR = std(SAR(:));

stat.mean_NSDR = mean(NSDR(:));
stat.std_NSDR = std(NSDR(:));
        

result.stat = stat;



