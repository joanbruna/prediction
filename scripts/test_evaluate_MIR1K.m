

params_aux = audio_config();

fs = params_aux.fs;
NFFT = params_aux.NFFT;
hop = params_aux.hop;

epsilon = 10;


FilePath='/misc/vlgscratch3/LecunGroup/pablo/MIR1K/Wavfile/';

files=dir([FilePath,'*.wav']);

epslion =10;

count = 1;
pt = 0;

for i=1:numel(files)
    
    
    if strncmp(files(i).name, 'abjones',7) || strncmp(files(i).name, 'amy',3) % ignore test
        continue; 
    end
    disp(files(i).name)
    

    [x,Fs] = audioread([FilePath,files(i).name]);
    
    wavinA= x(:,1);
    wavinE= x(:,2);
    mix=wavinA+wavinE;
    
    
    
    % mix
    mix = resample(mix,fs,Fs);
    
    S = compute_spectrum(mix,NFFT, hop);
    X = abs(S);
    
    X = softNormalize(X,epsilon);
    
    
    options.W = D_back;
    options.tau = 0.5;
    
    [A,theta,SA,An] = nmf_optflow_smooth(X,Dslow,options,ptheta);
    
    
    R = {};
    R{1} = Dslow* A;
    R{2} = D_back*An;
    
    y_out = wienerFilter2(R,S);
    y_out_of = y_out;
    
    m = length(y_out{1});
    x2 = wavinA(1:m);
    n2 = wavinE(1:m);
    
    
    
    [SDR,SIR,SAR,perm] = bss_eval_sources( mix', wavinE' );
    sdr_ = SDR(perm)';
    
    
    [SDR,SIR,SAR,perm] = bss_eval_sources( [y_out{1},y_out{2}]',[x2,n2]');
    
    
    NSDR=SDR(perm)'-sdr_;
    %     %%
    %     NSDR
    wavlength=length(wavinA);
    count=count+1;
    GNSDR(count)=NSDR(1);
    GNSDR_len(count)=wavlength;
    
    if pt
    xv = resample(wavinE,fs,Fs);
    Sv = compute_spectrum(xv,NFFT, hop);
    Xv = abs(Sv);
    Xv = softNormalize(Xv,epsilon);
        
    figure(2)
    subplot(311)
    dbimagesc(X+0.001);
    subplot(312)
    dbimagesc(Xv+0.001);
    subplot(313)
    dbimagesc(Dslow* A+0.001);
    end
    
    fprintf('NSDR voice: %f\nNSDR music: %f\n',NSDR(1),NSDR(2));
    %     fprintf('ifiel:%d\ncount:%d\nSDR:%f\nNSDR:%f\nSDR nets:%f\nNSDR nets:%f\nNSDR nets 1:%f\n',i, count,Parms.SDR,NSDR,SDR_n,NSDR_n,GNSDR_n1);


    
end
