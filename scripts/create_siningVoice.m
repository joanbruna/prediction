

params_aux = audio_config();

fs = params_aux.fs;
NFFT = params_aux.NFFT;
hop = params_aux.hop;

epsilon = 10;


FilePath='/misc/vlgscratch3/LecunGroup/pablo/MIR1K/Wavfile/';

files=dir([FilePath,'*.wav']);

XX_voice = [];
XX_back = [];

for i=1:numel(files)
    if strncmp(files(i).name, 'abjones',7) || strncmp(files(i).name, 'amy',3) % ignore test
        
        files(i).name
        
        [x,Fs] = audioread([FilePath,files(i).name]);
        x = resample(x,fs,Fs);
        
        S = compute_spectrum(x(:,2),NFFT, hop);
        X = abs(S);
        
        X = softNormalize(X,epsilon);
        
        XX_voice = [XX_voice X];
        
        % background
        
        S = compute_spectrum(x(:,1),NFFT, hop);
        X = abs(S);
        
        X = softNormalize(X,epsilon);
        
        XX_back = [XX_back X];
        
    end
    
end