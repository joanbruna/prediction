
function  result = separation_test_joan(testFun,test_files_1,test_files_2,options)

% SNR of the mixture - Always fixed
SNR_dB = getoptions(options,'SNR_dB',0);

% display results
verbose  = getoptions(options,'verbose',1);

% which audio files to use
id1 = getoptions(options,'id1',[1,1,1]);
id2 = getoptions(options,'id2',[1,1,1]);

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
    
    T = min(length(x1),length(x2));
    x1 = x1(1:T)/norm(x1(1:T));
    x2 = x2(1:T);
    
    if verbose
       fprintf('------------------------------\n')
       fprintf('File: %s, File %s\n\n',test_file_name_1,test_file_name_2)
    end
    
    x2 = x2/norm(x2)*power(10,(-SNR_dB)/20);
    mix = x1 + x2;
 
    TT=min(length(mix),options.N);
    mix = mix(1:TT);
    x1=x1(1:TT);
    x2=x2(1:TT);

    [~,~,speech1, speech2]=testFun(mix);

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



