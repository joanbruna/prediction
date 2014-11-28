

folderv = '/misc/vlgscratch3/LecunGroup/bruna/grid_data/Data_with_dev/';


test_file1 = 'female_test.wav';
test_file2 = 'male_test.wav';


%% Load trained network

%model_cnn = '/misc/vlgscratch3/LecunGroup/pablo/models/cnn/timit-cnn-512/';
model_cnn = '/misc/vlgscratch3/LecunGroup/pablo/models/cnn/timit-cnn-512-2layer/';

load([model_cnn 'net-epoch-195.mat'])

net_cnn.layers = net.layers(1:end-1);




%% RUN TEST

epsilon = 0.0001;
Fs = 16000;
NFFT = 1024;
hop = NFFT/2;

Npad_2 = 2^16;
options.N = Npad_2;

[x, fs] = audioread([folderv test_file1]);
x = resample(x,Fs,fs);
x1 = x'; T1 = length(x1);

[x, fs] = audioread([folderv test_file2]);
x = resample(x,Fs,fs);
fs = Fs;
x2 = x'; T2 = length(x2);

T = min([Npad_2,T1,T2]);

x1 = x1(1:T);
x2 = x2(1:T);

x1 = x1/norm(x1); 
x2 = x2/norm(x2);

mix = x1+x2;

is_stft = 1;
if is_stft
    X = compute_spectrum(mix,NFFT,hop);
    Xn = softNormalize(abs(X),epsilon);
end

[mask1,mask2] = cnn_demix(Xn,net_cnn);

% wiener filter included in the net
%     eps_1 = 1e-6;
%     V_ap = W1H1.^2 +W2H2.^2 + eps_1;
SPEECH1 = (mask1).*X;
SPEECH2 = (mask2).*X;

if is_stft
    speech1 = invert_spectrum(SPEECH1,NFFT,hop,T);
    speech2 = invert_spectrum(SPEECH2,NFFT,hop,T);
end


Parms =  BSS_EVAL(x1', x2', speech1(1:T)', speech2(1:T)', mix');

Parms

Parms2 =  BSS_EVAL_RNN(x1', x2', speech1(1:T)', speech2(1:T)', mix');

Parms2







