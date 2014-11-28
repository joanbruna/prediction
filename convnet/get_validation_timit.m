
function  valid = get_validation_timit(test_files_1,test_files_2,options)

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



for j=1:M
for i=1:N

    % get files
    x1 = test_files_1{id1(i,1)}{id1(i,2)}{id1(i,3)}.x;
    
    x2 = test_files_2{id2(j,1)}{id2(j,2)}{id2(j,3)}.x;
    
    T = min([length(x1),length(x2),Npad]);
    x1 = x1(1:T)/norm(x1(1:T));
    x2 = x2(1:T);
    
    x2 = x2/norm(x2)*power(10,(-SNR_dB)/20);
    
    x1_pad = pad_zeros(x1, Npad);
    x2_pad = pad_zeros(x2, Npad);
    
    
    mix = x1 + x2;

    %mix = (x1/norm(x1)+x2/norm(x2));
    if is_stft
        X = compute_spectrum(mix,NFFT,hop);
        Xn = softNormalize(abs(X),epsilon);
    end
    
    X

end
end


valid.x1 = x1;
valid.x2 = x2;
valid.mix = mix;
valid.X = X;


function out=pad_zeros(in, L)


out=zeros(1,L);
T=length(in);
if T > L
out = in(1+floor((T-L)/2):floor((T-L)/2)+L);
else

aux=zeros(1,L-T);
out(1:T)=in;
out(T+1:end)=aux;

end






